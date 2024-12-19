package stream

import (
	"fmt"
	"log/slog"
	"os"
	"sync"

	"github.com/Khan/genqlient/graphql"
	"github.com/wandb/wandb/core/internal/filestream"
	"github.com/wandb/wandb/core/internal/filetransfer"
	"github.com/wandb/wandb/core/internal/mailbox"
	"github.com/wandb/wandb/core/internal/observability"
	"github.com/wandb/wandb/core/internal/runfiles"
	"github.com/wandb/wandb/core/internal/runsummary"
	"github.com/wandb/wandb/core/internal/runwork"
	"github.com/wandb/wandb/core/internal/sentry_ext"
	"github.com/wandb/wandb/core/internal/settings"
	"github.com/wandb/wandb/core/internal/tensorboard"
	"github.com/wandb/wandb/core/internal/version"
	"github.com/wandb/wandb/core/internal/watcher"
	"github.com/wandb/wandb/core/internal/wboperation"
	"github.com/wandb/wandb/core/pkg/monitor"

	spb "github.com/wandb/wandb/core/pkg/service_go_proto"
)

const BufferSize = 32

// Stream processes incoming records for a single run.
//
// wandb consists of a service process (this code) to which one or more
// user processes connect (e.g. using the Python wandb library). The user
// processes send "records" to the service process that log to and modify
// the run, which the service process consumes asynchronously.
type Stream struct {
	// runWork is a channel of records to process.
	runWork runwork.RunWork

	// logger is the logger for the stream
	logger *observability.CoreLogger

	// wg is the WaitGroup for the stream
	wg sync.WaitGroup

	// settings is the settings for the stream
	settings *settings.Settings

	// handler is the handler for the stream
	handler *Handler

	// writer is the writer for the stream
	writer *Writer

	// sender is the sender for the stream
	sender *Sender

	// dispatcher is the dispatcher for the stream
	dispatcher *Dispatcher

	// sentryClient is the client used to report errors to sentry.io
	sentryClient *sentry_ext.Client
}

type StreamParams struct {
	Commit   string
	Settings *settings.Settings
	Sentry   *sentry_ext.Client
	LogLevel slog.Level
}

// NewStream creates a new stream with the given settings and responders.
func NewStream(
	params StreamParams,
) *Stream {
	operations := wboperation.NewOperations()

	writer, err := os.OpenFile(
		params.Settings.GetInternalLogFile(),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0666,
	)
	if err != nil {
		slog.Error(fmt.Sprintf("error opening log file: %s", err))
	}

	sentry := params.Sentry

	sentry.SetUser(
		params.Settings.GetEntity(),
		params.Settings.GetEmail(),
		params.Settings.GetUserName(),
	)

	logger := observability.NewCoreLogger(
		slog.New(slog.NewJSONHandler(
			writer,
			&slog.HandlerOptions{
				Level: params.LogLevel,
				// AddSource: true,
			},
		)),
		&observability.CoreLoggerParams{
			Tags: observability.Tags{
				"run_id":   params.Settings.GetRunID(),
				"run_url":  params.Settings.GetRunURL(),
				"project":  params.Settings.GetProject(),
				"base_url": params.Settings.GetBaseURL(),
			},
			Sentry: sentry,
		},
	)

	if params.Settings.GetSweepURL() != "" {
		logger.SetGlobalTags(observability.Tags{
			"sweep_url": params.Settings.GetSweepURL(),
		})
	}

	logger.Info("stream: starting", "core version", version.Version)

	s := &Stream{
		runWork:      runwork.New(BufferSize, logger),
		logger:       logger,
		settings:     params.Settings,
		sentryClient: params.Sentry,
	}

	// TODO: replace this with a logger that can be read by the user
	peeker := &observability.Peeker{}
	terminalPrinter := observability.NewPrinter()

	backendOrNil := NewBackend(s.logger, params.Settings)
	fileTransferStats := filetransfer.NewFileTransferStats()
	fileWatcher := watcher.New(watcher.Params{Logger: s.logger})
	tbHandler := tensorboard.NewTBHandler(tensorboard.Params{
		ExtraWork: s.runWork,
		Logger:    s.logger,
		Settings:  s.settings,
	})
	var graphqlClientOrNil graphql.Client
	var fileStreamOrNil filestream.FileStream
	var fileTransferManagerOrNil filetransfer.FileTransferManager
	var runfilesUploaderOrNil runfiles.Uploader
	if backendOrNil != nil {
		graphqlClientOrNil = NewGraphQLClient(backendOrNil, params.Settings, peeker)
		fileStreamOrNil = NewFileStream(
			backendOrNil,
			s.logger,
			operations,
			terminalPrinter,
			params.Settings,
			peeker,
		)
		fileTransferManagerOrNil = NewFileTransferManager(
			fileTransferStats,
			s.logger,
			params.Settings,
		)
		runfilesUploaderOrNil = NewRunfilesUploader(
			s.runWork,
			s.logger,
			operations,
			params.Settings,
			fileStreamOrNil,
			fileTransferManagerOrNil,
			fileWatcher,
			graphqlClientOrNil,
		)
	}

	mailbox := mailbox.New()

	s.handler = NewHandler(
		HandlerParams{
			Logger:            s.logger,
			Operations:        operations,
			Settings:          s.settings,
			FwdChan:           make(chan runwork.Work, BufferSize),
			OutChan:           make(chan *spb.Result, BufferSize),
			SystemMonitor:     monitor.NewSystemMonitor(s.logger, s.settings, s.runWork),
			TBHandler:         tbHandler,
			FileTransferStats: fileTransferStats,
			Mailbox:           mailbox,
			TerminalPrinter:   terminalPrinter,
			Commit:            params.Commit,
		},
	)

	s.writer = NewWriter(
		WriterParams{
			Logger:   s.logger,
			Settings: s.settings,
			FwdChan:  make(chan runwork.Work, BufferSize),
		},
	)

	s.sender = NewSender(
		s.runWork,
		SenderParams{
			Logger:              s.logger,
			Operations:          operations,
			Settings:            s.settings,
			Backend:             backendOrNil,
			FileStream:          fileStreamOrNil,
			FileTransferManager: fileTransferManagerOrNil,
			FileTransferStats:   fileTransferStats,
			FileWatcher:         fileWatcher,
			RunfilesUploader:    runfilesUploaderOrNil,
			TBHandler:           tbHandler,
			Peeker:              peeker,
			RunSummary:          runsummary.New(),
			GraphqlClient:       graphqlClientOrNil,
			OutChan:             make(chan *spb.Result, BufferSize),
			Mailbox:             mailbox,
		},
	)

	s.dispatcher = NewDispatcher(s.logger)

	s.logger.Info("created new stream", "id", s.settings.GetRunID())
	return s
}

// AddResponders adds the given responders to the stream's dispatcher.
func (s *Stream) AddResponders(entries ...ResponderEntry) {
	s.dispatcher.AddResponders(entries...)
}

// UpdateSettings updates the stream's settings with the given settings.
func (s *Stream) UpdateSettings(newSettings *settings.Settings) {
	s.settings = newSettings
}

// GetSettings returns the stream's settings.
func (s *Stream) GetSettings() *settings.Settings {
	return s.settings
}

// UpdateRunURLTag updates the run URL tag in the stream's logger.
// TODO: this should be removed when we remove informStart.
func (s *Stream) UpdateRunURLTag() {
	s.logger.SetGlobalTags(observability.Tags{
		"run_url": s.settings.GetRunURL(),
	})
}

// Start starts the stream's handler, writer, sender, and dispatcher.
// We use Stream's wait group to ensure that all of these components are cleanly
// finalized and closed when the stream is closed in Stream.Close().
func (s *Stream) Start() {
	// handle the client requests with the handler
	s.wg.Add(1)
	go func() {
		s.handler.Do(s.runWork.Chan())
		s.wg.Done()
	}()

	// write the data to a transaction log
	s.wg.Add(1)
	go func() {
		s.writer.Do(s.handler.fwdChan)
		s.wg.Done()
	}()

	// send the data to the server
	s.wg.Add(1)
	go func() {
		s.sender.Do(s.writer.fwdChan)
		s.wg.Done()
	}()

	// handle dispatching between components
	for _, ch := range []chan *spb.Result{s.handler.outChan, s.sender.outChan} {
		s.wg.Add(1)
		go func(ch chan *spb.Result) {
			for result := range ch {
				s.dispatcher.handleRespond(result)
			}
			s.wg.Done()
		}(ch)
	}

	s.logger.Info("stream: started", "id", s.settings.GetRunID())
}

// HandleRecord handles the given record by sending it to the stream's handler.
func (s *Stream) HandleRecord(rec *spb.Record) {
	s.logger.Debug("handling record", "record", rec)
	s.runWork.AddWork(runwork.WorkFromRecord(rec))
}

// Close waits for all run messages to be fully processed.
func (s *Stream) Close() {
	s.logger.Info("stream: closing", "id", s.settings.GetRunID())
	s.runWork.Close()
	s.wg.Wait()
	s.logger.Info("stream: closed", "id", s.settings.GetRunID())
}

// FinishAndClose emits an exit record, waits for all run messages
// to be fully processed, and prints the run footer to the terminal.
func (s *Stream) FinishAndClose(exitCode int32) {
	if !s.settings.IsSync() {
		s.HandleRecord(&spb.Record{
			RecordType: &spb.Record_Exit{
				Exit: &spb.RunExitRecord{
					ExitCode: exitCode,
				}},
			Control: &spb.Control{AlwaysSend: true},
		})
	}

	s.Close()

	if s.settings.IsOffline() {
		PrintFooterOffline(s.settings.Proto)
	} else {
		run := s.handler.GetRun()
		PrintFooterOnline(run, s.settings.Proto)
	}
}
