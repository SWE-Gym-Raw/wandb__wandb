package server

import (
	"context"
	"log/slog"
	"os"
	"sync"

	"github.com/wandb/wandb/core/pkg/observability"
	"github.com/wandb/wandb/core/pkg/service"
)

type WriterOption func(*Writer)

func WithWriterFwdChannel(fwd chan *service.Record) WriterOption {
	return func(w *Writer) {
		w.fwdChan = fwd
	}
}

func WithWriterSettings(settings *service.Settings) WriterOption {
	return func(w *Writer) {
		w.settings = settings
	}
}

// Writer is responsible for writing messages to the append-only log.
// It receives messages from the handler, processes them,
// if the message is to be persisted it writes them to the log.
// It also sends the messages to the sender.
type Writer struct {
	// ctx is the context for the writer
	ctx context.Context

	// settings is the settings for the writer
	settings *service.Settings

	// logger is the logger for the writer
	logger *observability.CoreLogger

	// fwdChan is the channel for forwarding messages to the sender
	fwdChan chan *service.Record

	// storeChan is the channel for messages to be stored
	storeChan chan *service.Record

	// store is the store for the writer
	store *Store

	// recordNum is the running count of stored records
	recordNum int64

	// wg is the wait group for the writer
	wg sync.WaitGroup
}

// NewWriter returns a new Writer
func NewWriter(ctx context.Context, logger *observability.CoreLogger, opts ...WriterOption) *Writer {
	w := &Writer{
		ctx:    ctx,
		logger: observability.NewCoreLogger(slog.New(logger.Handler().WithGroup("writer"))),
		wg:     sync.WaitGroup{},
	}
	for _, opt := range opts {
		opt(w)
	}
	return w
}

func (w *Writer) startStore() {
	if w.settings.GetXSync().GetValue() {
		// do not set up store if we are syncing an offline run
		return
	}

	w.storeChan = make(chan *service.Record, BufferSize*8)

	var err error
	w.store = NewStore(w.ctx, w.settings.GetSyncFile().GetValue(), w.logger)
	err = w.store.Open(os.O_WRONLY)
	if err != nil {
		w.logger.Error("failed to open store", "error", err)
		panic(err)
		// w.logger.CaptureFatalAndPanic("writer: error creating store", err)
	}

	w.wg.Add(1)
	go func() {
		for record := range w.storeChan {
			if err = w.store.Write(record); err != nil {
				w.logger.Error("failed to write to store", "error", err)
			}
		}

		if err = w.store.Close(); err != nil {
			w.logger.Error("failed to close store", "error", err)
			// w.logger.CaptureError("writer: error closing store", err)
		}
		w.wg.Done()
	}()
}

// do is the main loop of the writer to process incoming messages
func (w *Writer) Do(inChan <-chan *service.Record) {
	defer w.logger.Reraise()
	w.logger.Info("started writer")

	w.startStore()

	for record := range inChan {
		w.logger.Debug("received record", "record", record.RecordType)
		w.handleRecord(record)
	}
	w.Close()
	w.wg.Wait()
}

// Close closes the writer and all its resources
// which includes the store
func (w *Writer) Close() {
	close(w.fwdChan)
	if w.storeChan != nil {
		close(w.storeChan)
	}
	w.logger.Info("closed writer")
}

// handleRecord Writing messages to the append-only log,
// and passing them to the sender.
// We ensure that the messages are written to the log
// before they are sent to the server.
func (w *Writer) handleRecord(record *service.Record) {
	switch record.RecordType.(type) {
	case *service.Record_Request:
		w.sendRecord(record)
	case nil:
		w.logger.Error("nil record type")
	default:
		w.sendRecord(record)
		w.storeRecord(record)
	}
}

// storeRecord stores the record in the append-only log
func (w *Writer) storeRecord(record *service.Record) {
	if record.GetControl().GetLocal() {
		return
	}
	w.recordNum += 1
	record.Num = w.recordNum
	w.storeChan <- record
}

func (w *Writer) sendRecord(record *service.Record) {
	// TODO: redo it so it only uses control
	if w.settings.GetXOffline().GetValue() && !record.GetControl().GetAlwaysSend() {
		return
	}
	w.fwdChan <- record
}
