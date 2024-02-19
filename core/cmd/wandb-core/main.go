package main

import (
	"context"
	"flag"
	"log/slog"
	_ "net/http/pprof"
	"os"
	"runtime"
	"runtime/trace"

	"github.com/getsentry/sentry-go"
	"github.com/wandb/wandb/core/pkg/observability"
	"github.com/wandb/wandb/core/pkg/server"
)

// this is set by the build script and used by the observability package
var commit string

func init() {
	runtime.SetBlockProfileRate(1)
}

func main() {
	// Flags to control the server
	portFilename := flag.String("port-filename", "port_file.txt", "filename for port to communicate with client")
	pid := flag.Int("pid", 0, "pid of the process to communicate with")
	debugLevel := flag.Bool("debug", false, "enable debug logging")
	disableAnalytics := flag.Bool("no-observability", false, "turn off observability")
	traceFile := flag.String("trace", "", "file name to write trace output to")
	// todo: remove these flags, they are here for backward compatibility
	serveSock := flag.Bool("serve-sock", false, "use sockets")

	flag.Parse()

	// set up sentry reporting
	observability.InitSentry(*disableAnalytics, commit)
	defer sentry.Flush(2)

	// store commit hash in context
	ctx := context.Background()
	ctx = context.WithValue(ctx, observability.Commit("commit"), commit)

	var loggerPath string
	if file, _ := observability.GetLoggerPath(); file != nil {
		level := slog.LevelInfo
		if *debugLevel {
			level = slog.LevelDebug
		}
		opts := &slog.HandlerOptions{
			Level:     level,
			AddSource: true,
		}
		logger := slog.New(slog.NewJSONHandler(file, opts))
		slog.SetDefault(logger)
		logger.LogAttrs(
			ctx,
			slog.LevelInfo,
			"started logging, with flags",
			slog.String("fname", *portFilename),
			slog.Int("pid", *pid),
			slog.Bool("debug", *debugLevel),
			slog.Bool("noAnalytics", *disableAnalytics),
			slog.Bool("serveSock", *serveSock),
		)
		loggerPath = file.Name()
		defer file.Close()
	}

	// TODO: replace it with a flag
	if *traceFile != "" {
		f, err := os.Create(*traceFile)
		if err != nil {
			slog.Error("failed to create trace output file", "err", err)
			panic(err)
		}
		defer func() {
			if err = f.Close(); err != nil {
				slog.Error("failed to close trace file", "err", err)
			}
		}()

		if err = trace.Start(f); err != nil {
			slog.Error("failed to start trace", "err", err)
			panic(err)
		}
		defer trace.Stop()
	}
	serve := server.NewServer(ctx, "127.0.0.1:0", *portFilename)
	serve.SetDefaultLoggerPath(loggerPath)
	serve.Close()
}
