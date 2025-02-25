package filetransfer

import (
	"github.com/hashicorp/go-retryablehttp"
	"github.com/wandb/wandb/core/internal/observability"
)

type FileTransfer interface {
	Upload(task *DefaultUploadTask) error
	Download(task *DefaultDownloadTask) error
}

type ArtifactFileTransfer interface {
	Upload(task *DefaultUploadTask) error
	Download(task *ReferenceArtifactDownloadTask) error
}

// FileTransfers is a collection of file transfers by upload destination type.
type FileTransfers struct {
	// Default makes an HTTP request to the destination URL with the file contents.
	Default FileTransfer

	// GCS connects to GCloud to upload/download files given their paths
	GCS ArtifactFileTransfer

	// S3 connects to AWS to upload/download files given their paths
	S3 ArtifactFileTransfer

	// Azure connects to Azure to upload/download files given their paths
	Azure ArtifactFileTransfer
}

// NewFileTransfers creates a new fileTransfers
func NewFileTransfers(
	client *retryablehttp.Client,
	logger *observability.CoreLogger,
	fileTransferStats FileTransferStats,
) *FileTransfers {
	defaultFileTransfer := NewDefaultFileTransfer(client, logger, fileTransferStats)
	gcsFileTransfer := NewGCSFileTransfer(nil, logger, fileTransferStats)
	s3FileTransfer := NewS3FileTransfer(nil, logger, fileTransferStats)
	azureFileTransfer := NewAzureFileTransfer(nil, logger, fileTransferStats)

	return &FileTransfers{
		Default: defaultFileTransfer,
		GCS:     gcsFileTransfer,
		S3:      s3FileTransfer,
		Azure:   azureFileTransfer,
	}
}
