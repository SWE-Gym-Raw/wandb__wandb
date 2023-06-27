# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: wandb/proto/wandb_internal.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from wandb.proto import wandb_base_pb2 as wandb_dot_proto_dot_wandb__base__pb2
from wandb.proto import wandb_telemetry_pb2 as wandb_dot_proto_dot_wandb__telemetry__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n wandb/proto/wandb_internal.proto\x12\x0ewandb_internal\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1cwandb/proto/wandb_base.proto\x1a!wandb/proto/wandb_telemetry.proto\"\x9c\t\n\x06Record\x12\x0b\n\x03num\x18\x01 \x01(\x03\x12\x30\n\x07history\x18\x02 \x01(\x0b\x32\x1d.wandb_internal.HistoryRecordH\x00\x12\x30\n\x07summary\x18\x03 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecordH\x00\x12.\n\x06output\x18\x04 \x01(\x0b\x32\x1c.wandb_internal.OutputRecordH\x00\x12.\n\x06\x63onfig\x18\x05 \x01(\x0b\x32\x1c.wandb_internal.ConfigRecordH\x00\x12,\n\x05\x66iles\x18\x06 \x01(\x0b\x32\x1b.wandb_internal.FilesRecordH\x00\x12,\n\x05stats\x18\x07 \x01(\x0b\x32\x1b.wandb_internal.StatsRecordH\x00\x12\x32\n\x08\x61rtifact\x18\x08 \x01(\x0b\x32\x1e.wandb_internal.ArtifactRecordH\x00\x12,\n\x08tbrecord\x18\t \x01(\x0b\x32\x18.wandb_internal.TBRecordH\x00\x12,\n\x05\x61lert\x18\n \x01(\x0b\x32\x1b.wandb_internal.AlertRecordH\x00\x12\x34\n\ttelemetry\x18\x0b \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecordH\x00\x12.\n\x06metric\x18\x0c \x01(\x0b\x32\x1c.wandb_internal.MetricRecordH\x00\x12\x35\n\noutput_raw\x18\r \x01(\x0b\x32\x1f.wandb_internal.OutputRawRecordH\x00\x12(\n\x03run\x18\x11 \x01(\x0b\x32\x19.wandb_internal.RunRecordH\x00\x12-\n\x04\x65xit\x18\x12 \x01(\x0b\x32\x1d.wandb_internal.RunExitRecordH\x00\x12,\n\x05\x66inal\x18\x14 \x01(\x0b\x32\x1b.wandb_internal.FinalRecordH\x00\x12.\n\x06header\x18\x15 \x01(\x0b\x32\x1c.wandb_internal.HeaderRecordH\x00\x12.\n\x06\x66ooter\x18\x16 \x01(\x0b\x32\x1c.wandb_internal.FooterRecordH\x00\x12\x39\n\npreempting\x18\x17 \x01(\x0b\x32#.wandb_internal.RunPreemptingRecordH\x00\x12;\n\rlink_artifact\x18\x18 \x01(\x0b\x32\".wandb_internal.LinkArtifactRecordH\x00\x12\x39\n\x0cuse_artifact\x18\x19 \x01(\x0b\x32!.wandb_internal.UseArtifactRecordH\x00\x12*\n\x07request\x18\x64 \x01(\x0b\x32\x17.wandb_internal.RequestH\x00\x12(\n\x07\x63ontrol\x18\x10 \x01(\x0b\x32\x17.wandb_internal.Control\x12\x0c\n\x04uuid\x18\x13 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfoB\r\n\x0brecord_type\"\x91\x01\n\x07\x43ontrol\x12\x10\n\x08req_resp\x18\x01 \x01(\x08\x12\r\n\x05local\x18\x02 \x01(\x08\x12\x10\n\x08relay_id\x18\x03 \x01(\t\x12\x14\n\x0cmailbox_slot\x18\x04 \x01(\t\x12\x13\n\x0b\x61lways_send\x18\x05 \x01(\x08\x12\x14\n\x0c\x66low_control\x18\x06 \x01(\x08\x12\x12\n\nend_offset\x18\x07 \x01(\x03\"\xf3\x03\n\x06Result\x12\x35\n\nrun_result\x18\x11 \x01(\x0b\x32\x1f.wandb_internal.RunUpdateResultH\x00\x12\x34\n\x0b\x65xit_result\x18\x12 \x01(\x0b\x32\x1d.wandb_internal.RunExitResultH\x00\x12\x33\n\nlog_result\x18\x14 \x01(\x0b\x32\x1d.wandb_internal.HistoryResultH\x00\x12\x37\n\x0esummary_result\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.SummaryResultH\x00\x12\x35\n\routput_result\x18\x16 \x01(\x0b\x32\x1c.wandb_internal.OutputResultH\x00\x12\x35\n\rconfig_result\x18\x17 \x01(\x0b\x32\x1c.wandb_internal.ConfigResultH\x00\x12,\n\x08response\x18\x64 \x01(\x0b\x32\x18.wandb_internal.ResponseH\x00\x12(\n\x07\x63ontrol\x18\x10 \x01(\x0b\x32\x17.wandb_internal.Control\x12\x0c\n\x04uuid\x18\x18 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._ResultInfoB\r\n\x0bresult_type\":\n\x0b\x46inalRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\";\n\x0cHeaderRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\";\n\x0c\x46ooterRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\xe0\x04\n\tRunRecord\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0e\n\x06\x65ntity\x18\x02 \x01(\t\x12\x0f\n\x07project\x18\x03 \x01(\t\x12,\n\x06\x63onfig\x18\x04 \x01(\x0b\x32\x1c.wandb_internal.ConfigRecord\x12.\n\x07summary\x18\x05 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecord\x12\x11\n\trun_group\x18\x06 \x01(\t\x12\x10\n\x08job_name\x18\x07 \x01(\t\x12\x10\n\x08job_type\x18\x08 \x01(\t\x12\x14\n\x0c\x64isplay_name\x18\t \x01(\t\x12\r\n\x05notes\x18\n \x01(\t\x12\x0c\n\x04tags\x18\x0b \x03(\t\x12\x30\n\x08settings\x18\x0c \x01(\x0b\x32\x1e.wandb_internal.SettingsRecord\x12\x10\n\x08sweep_id\x18\r \x01(\t\x12\x0c\n\x04host\x18\x0e \x01(\t\x12\x15\n\rstarting_step\x18\x0f \x01(\x03\x12\x12\n\nstorage_id\x18\x10 \x01(\t\x12.\n\nstart_time\x18\x11 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0f\n\x07resumed\x18\x12 \x01(\x08\x12\x32\n\ttelemetry\x18\x13 \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecord\x12\x0f\n\x07runtime\x18\x14 \x01(\x05\x12*\n\x03git\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.GitRepoRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"3\n\rGitRepoRecord\x12\x12\n\nremote_url\x18\x01 \x01(\t\x12\x0e\n\x06\x63ommit\x18\x02 \x01(\t\"c\n\x0fRunUpdateResult\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"\xac\x01\n\tErrorInfo\x12\x0f\n\x07message\x18\x01 \x01(\t\x12\x31\n\x04\x63ode\x18\x02 \x01(\x0e\x32#.wandb_internal.ErrorInfo.ErrorCode\"[\n\tErrorCode\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x11\n\rCOMMUNICATION\x10\x01\x12\x12\n\x0e\x41UTHENTICATION\x10\x02\x12\t\n\x05USAGE\x10\x03\x12\x0f\n\x0bUNSUPPORTED\x10\x04\"`\n\rRunExitRecord\x12\x11\n\texit_code\x18\x01 \x01(\x05\x12\x0f\n\x07runtime\x18\x02 \x01(\x05\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x0f\n\rRunExitResult\"B\n\x13RunPreemptingRecord\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x15\n\x13RunPreemptingResult\"i\n\x0eSettingsRecord\x12*\n\x04item\x18\x01 \x03(\x0b\x32\x1c.wandb_internal.SettingsItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"/\n\x0cSettingsItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x1a\n\x0bHistoryStep\x12\x0b\n\x03num\x18\x01 \x01(\x03\"\x92\x01\n\rHistoryRecord\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.HistoryItem\x12)\n\x04step\x18\x02 \x01(\x0b\x32\x1b.wandb_internal.HistoryStep\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\x0bHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0f\n\rHistoryResult\"\xdc\x01\n\x0cOutputRecord\x12<\n\x0boutput_type\x18\x01 \x01(\x0e\x32\'.wandb_internal.OutputRecord.OutputType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04line\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"$\n\nOutputType\x12\n\n\x06STDERR\x10\x00\x12\n\n\x06STDOUT\x10\x01\"\x0e\n\x0cOutputResult\"\xe2\x01\n\x0fOutputRawRecord\x12?\n\x0boutput_type\x18\x01 \x01(\x0e\x32*.wandb_internal.OutputRawRecord.OutputType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x0c\n\x04line\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"$\n\nOutputType\x12\n\n\x06STDERR\x10\x00\x12\n\n\x06STDOUT\x10\x01\"\x11\n\x0fOutputRawResult\"\x98\x03\n\x0cMetricRecord\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\tglob_name\x18\x02 \x01(\t\x12\x13\n\x0bstep_metric\x18\x04 \x01(\t\x12\x19\n\x11step_metric_index\x18\x05 \x01(\x05\x12.\n\x07options\x18\x06 \x01(\x0b\x32\x1d.wandb_internal.MetricOptions\x12.\n\x07summary\x18\x07 \x01(\x0b\x32\x1d.wandb_internal.MetricSummary\x12\x35\n\x04goal\x18\x08 \x01(\x0e\x32\'.wandb_internal.MetricRecord.MetricGoal\x12/\n\x08_control\x18\t \x01(\x0b\x32\x1d.wandb_internal.MetricControl\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\nMetricGoal\x12\x0e\n\nGOAL_UNSET\x10\x00\x12\x11\n\rGOAL_MINIMIZE\x10\x01\x12\x11\n\rGOAL_MAXIMIZE\x10\x02\"\x0e\n\x0cMetricResult\"C\n\rMetricOptions\x12\x11\n\tstep_sync\x18\x01 \x01(\x08\x12\x0e\n\x06hidden\x18\x02 \x01(\x08\x12\x0f\n\x07\x64\x65\x66ined\x18\x03 \x01(\x08\"\"\n\rMetricControl\x12\x11\n\toverwrite\x18\x01 \x01(\x08\"o\n\rMetricSummary\x12\x0b\n\x03min\x18\x01 \x01(\x08\x12\x0b\n\x03max\x18\x02 \x01(\x08\x12\x0c\n\x04mean\x18\x03 \x01(\x08\x12\x0c\n\x04\x62\x65st\x18\x04 \x01(\x08\x12\x0c\n\x04last\x18\x05 \x01(\x08\x12\x0c\n\x04none\x18\x06 \x01(\x08\x12\x0c\n\x04\x63opy\x18\x07 \x01(\x08\"\x93\x01\n\x0c\x43onfigRecord\x12*\n\x06update\x18\x01 \x03(\x0b\x32\x1a.wandb_internal.ConfigItem\x12*\n\x06remove\x18\x02 \x03(\x0b\x32\x1a.wandb_internal.ConfigItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"A\n\nConfigItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0e\n\x0c\x43onfigResult\"\x96\x01\n\rSummaryRecord\x12+\n\x06update\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\x12+\n\x06remove\x18\x02 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"B\n\x0bSummaryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\x0f\n\rSummaryResult\"d\n\x0b\x46ilesRecord\x12(\n\x05\x66iles\x18\x01 \x03(\x0b\x32\x19.wandb_internal.FilesItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x90\x01\n\tFilesItem\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x34\n\x06policy\x18\x02 \x01(\x0e\x32$.wandb_internal.FilesItem.PolicyType\x12\x15\n\rexternal_path\x18\x10 \x01(\t\"(\n\nPolicyType\x12\x07\n\x03NOW\x10\x00\x12\x07\n\x03\x45ND\x10\x01\x12\x08\n\x04LIVE\x10\x02\"\r\n\x0b\x46ilesResult\"\xe6\x01\n\x0bStatsRecord\x12\x39\n\nstats_type\x18\x01 \x01(\x0e\x32%.wandb_internal.StatsRecord.StatsType\x12-\n\ttimestamp\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\'\n\x04item\x18\x03 \x03(\x0b\x32\x19.wandb_internal.StatsItem\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x17\n\tStatsType\x12\n\n\x06SYSTEM\x10\x00\",\n\tStatsItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x10 \x01(\t\"\xaa\x03\n\x0e\x41rtifactRecord\x12\x0e\n\x06run_id\x18\x01 \x01(\t\x12\x0f\n\x07project\x18\x02 \x01(\t\x12\x0e\n\x06\x65ntity\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\x0c\n\x04name\x18\x05 \x01(\t\x12\x0e\n\x06\x64igest\x18\x06 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x07 \x01(\t\x12\x10\n\x08metadata\x18\x08 \x01(\t\x12\x14\n\x0cuser_created\x18\t \x01(\x08\x12\x18\n\x10use_after_commit\x18\n \x01(\x08\x12\x0f\n\x07\x61liases\x18\x0b \x03(\t\x12\x32\n\x08manifest\x18\x0c \x01(\x0b\x32 .wandb_internal.ArtifactManifest\x12\x16\n\x0e\x64istributed_id\x18\r \x01(\t\x12\x10\n\x08\x66inalize\x18\x0e \x01(\x08\x12\x11\n\tclient_id\x18\x0f \x01(\t\x12\x1a\n\x12sequence_client_id\x18\x10 \x01(\t\x12\x19\n\x11incremental_beta1\x18\x64 \x01(\x08\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\xbc\x01\n\x10\x41rtifactManifest\x12\x0f\n\x07version\x18\x01 \x01(\x05\x12\x16\n\x0estorage_policy\x18\x02 \x01(\t\x12\x46\n\x15storage_policy_config\x18\x03 \x03(\x0b\x32\'.wandb_internal.StoragePolicyConfigItem\x12\x37\n\x08\x63ontents\x18\x04 \x03(\x0b\x32%.wandb_internal.ArtifactManifestEntry\"\xbb\x01\n\x15\x41rtifactManifestEntry\x12\x0c\n\x04path\x18\x01 \x01(\t\x12\x0e\n\x06\x64igest\x18\x02 \x01(\t\x12\x0b\n\x03ref\x18\x03 \x01(\t\x12\x0c\n\x04size\x18\x04 \x01(\x03\x12\x10\n\x08mimetype\x18\x05 \x01(\t\x12\x12\n\nlocal_path\x18\x06 \x01(\t\x12\x19\n\x11\x62irth_artifact_id\x18\x07 \x01(\t\x12(\n\x05\x65xtra\x18\x10 \x03(\x0b\x32\x19.wandb_internal.ExtraItem\",\n\tExtraItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x02 \x01(\t\":\n\x17StoragePolicyConfigItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nvalue_json\x18\x02 \x01(\t\"\x10\n\x0e\x41rtifactResult\"\x14\n\x12LinkArtifactResult\"\xcf\x01\n\x12LinkArtifactRecord\x12\x11\n\tclient_id\x18\x01 \x01(\t\x12\x11\n\tserver_id\x18\x02 \x01(\t\x12\x16\n\x0eportfolio_name\x18\x03 \x01(\t\x12\x18\n\x10portfolio_entity\x18\x04 \x01(\t\x12\x19\n\x11portfolio_project\x18\x05 \x01(\t\x12\x19\n\x11portfolio_aliases\x18\x06 \x03(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"h\n\x08TBRecord\x12\x0f\n\x07log_dir\x18\x01 \x01(\t\x12\x0c\n\x04save\x18\x02 \x01(\x08\x12\x10\n\x08root_dir\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\n\n\x08TBResult\"}\n\x0b\x41lertRecord\x12\r\n\x05title\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t\x12\r\n\x05level\x18\x03 \x01(\t\x12\x15\n\rwait_duration\x18\x04 \x01(\x03\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\r\n\x0b\x41lertResult\"\x91\r\n\x07Request\x12\x38\n\x0bstop_status\x18\x01 \x01(\x0b\x32!.wandb_internal.StopStatusRequestH\x00\x12>\n\x0enetwork_status\x18\x02 \x01(\x0b\x32$.wandb_internal.NetworkStatusRequestH\x00\x12-\n\x05\x64\x65\x66\x65r\x18\x03 \x01(\x0b\x32\x1c.wandb_internal.DeferRequestH\x00\x12\x38\n\x0bget_summary\x18\x04 \x01(\x0b\x32!.wandb_internal.GetSummaryRequestH\x00\x12-\n\x05login\x18\x05 \x01(\x0b\x32\x1c.wandb_internal.LoginRequestH\x00\x12-\n\x05pause\x18\x06 \x01(\x0b\x32\x1c.wandb_internal.PauseRequestH\x00\x12/\n\x06resume\x18\x07 \x01(\x0b\x32\x1d.wandb_internal.ResumeRequestH\x00\x12\x34\n\tpoll_exit\x18\x08 \x01(\x0b\x32\x1f.wandb_internal.PollExitRequestH\x00\x12@\n\x0fsampled_history\x18\t \x01(\x0b\x32%.wandb_internal.SampledHistoryRequestH\x00\x12@\n\x0fpartial_history\x18\n \x01(\x0b\x32%.wandb_internal.PartialHistoryRequestH\x00\x12\x34\n\trun_start\x18\x0b \x01(\x0b\x32\x1f.wandb_internal.RunStartRequestH\x00\x12<\n\rcheck_version\x18\x0c \x01(\x0b\x32#.wandb_internal.CheckVersionRequestH\x00\x12:\n\x0clog_artifact\x18\r \x01(\x0b\x32\".wandb_internal.LogArtifactRequestH\x00\x12<\n\rartifact_send\x18\x0e \x01(\x0b\x32#.wandb_internal.ArtifactSendRequestH\x00\x12<\n\rartifact_poll\x18\x0f \x01(\x0b\x32#.wandb_internal.ArtifactPollRequestH\x00\x12<\n\rartifact_done\x18\x10 \x01(\x0b\x32#.wandb_internal.ArtifactDoneRequestH\x00\x12\x35\n\tkeepalive\x18\x11 \x01(\x0b\x32 .wandb_internal.KeepaliveRequestH\x00\x12\x36\n\nrun_status\x18\x14 \x01(\x0b\x32 .wandb_internal.RunStatusRequestH\x00\x12/\n\x06\x63\x61ncel\x18\x15 \x01(\x0b\x32\x1d.wandb_internal.CancelRequestH\x00\x12\x33\n\x08shutdown\x18@ \x01(\x0b\x32\x1f.wandb_internal.ShutdownRequestH\x00\x12/\n\x06\x61ttach\x18\x41 \x01(\x0b\x32\x1d.wandb_internal.AttachRequestH\x00\x12/\n\x06status\x18\x42 \x01(\x0b\x32\x1d.wandb_internal.StatusRequestH\x00\x12\x38\n\x0bserver_info\x18\x43 \x01(\x0b\x32!.wandb_internal.ServerInfoRequestH\x00\x12\x38\n\x0bsender_mark\x18\x44 \x01(\x0b\x32!.wandb_internal.SenderMarkRequestH\x00\x12\x38\n\x0bsender_read\x18\x45 \x01(\x0b\x32!.wandb_internal.SenderReadRequestH\x00\x12<\n\rstatus_report\x18\x46 \x01(\x0b\x32#.wandb_internal.StatusReportRequestH\x00\x12>\n\x0esummary_record\x18G \x01(\x0b\x32$.wandb_internal.SummaryRecordRequestH\x00\x12\x42\n\x10telemetry_record\x18H \x01(\x0b\x32&.wandb_internal.TelemetryRecordRequestH\x00\x12\x39\n\x0btest_inject\x18\xe8\x07 \x01(\x0b\x32!.wandb_internal.TestInjectRequestH\x00\x42\x0e\n\x0crequest_type\"\x8c\n\n\x08Response\x12?\n\x12keepalive_response\x18\x12 \x01(\x0b\x32!.wandb_internal.KeepaliveResponseH\x00\x12\x42\n\x14stop_status_response\x18\x13 \x01(\x0b\x32\".wandb_internal.StopStatusResponseH\x00\x12H\n\x17network_status_response\x18\x14 \x01(\x0b\x32%.wandb_internal.NetworkStatusResponseH\x00\x12\x37\n\x0elogin_response\x18\x18 \x01(\x0b\x32\x1d.wandb_internal.LoginResponseH\x00\x12\x42\n\x14get_summary_response\x18\x19 \x01(\x0b\x32\".wandb_internal.GetSummaryResponseH\x00\x12>\n\x12poll_exit_response\x18\x1a \x01(\x0b\x32 .wandb_internal.PollExitResponseH\x00\x12J\n\x18sampled_history_response\x18\x1b \x01(\x0b\x32&.wandb_internal.SampledHistoryResponseH\x00\x12>\n\x12run_start_response\x18\x1c \x01(\x0b\x32 .wandb_internal.RunStartResponseH\x00\x12\x46\n\x16\x63heck_version_response\x18\x1d \x01(\x0b\x32$.wandb_internal.CheckVersionResponseH\x00\x12\x44\n\x15log_artifact_response\x18\x1e \x01(\x0b\x32#.wandb_internal.LogArtifactResponseH\x00\x12\x46\n\x16\x61rtifact_send_response\x18\x1f \x01(\x0b\x32$.wandb_internal.ArtifactSendResponseH\x00\x12\x46\n\x16\x61rtifact_poll_response\x18  \x01(\x0b\x32$.wandb_internal.ArtifactPollResponseH\x00\x12@\n\x13run_status_response\x18# \x01(\x0b\x32!.wandb_internal.RunStatusResponseH\x00\x12\x39\n\x0f\x63\x61ncel_response\x18$ \x01(\x0b\x32\x1e.wandb_internal.CancelResponseH\x00\x12=\n\x11shutdown_response\x18@ \x01(\x0b\x32 .wandb_internal.ShutdownResponseH\x00\x12\x39\n\x0f\x61ttach_response\x18\x41 \x01(\x0b\x32\x1e.wandb_internal.AttachResponseH\x00\x12\x39\n\x0fstatus_response\x18\x42 \x01(\x0b\x32\x1e.wandb_internal.StatusResponseH\x00\x12\x42\n\x14server_info_response\x18\x43 \x01(\x0b\x32\".wandb_internal.ServerInfoResponseH\x00\x12\x43\n\x14test_inject_response\x18\xe8\x07 \x01(\x0b\x32\".wandb_internal.TestInjectResponseH\x00\x42\x0f\n\rresponse_type\"\xc0\x02\n\x0c\x44\x65\x66\x65rRequest\x12\x36\n\x05state\x18\x01 \x01(\x0e\x32\'.wandb_internal.DeferRequest.DeferState\"\xf7\x01\n\nDeferState\x12\t\n\x05\x42\x45GIN\x10\x00\x12\r\n\tFLUSH_RUN\x10\x01\x12\x0f\n\x0b\x46LUSH_STATS\x10\x02\x12\x19\n\x15\x46LUSH_PARTIAL_HISTORY\x10\x03\x12\x0c\n\x08\x46LUSH_TB\x10\x04\x12\r\n\tFLUSH_SUM\x10\x05\x12\x13\n\x0f\x46LUSH_DEBOUNCER\x10\x06\x12\x10\n\x0c\x46LUSH_OUTPUT\x10\x07\x12\r\n\tFLUSH_JOB\x10\x08\x12\r\n\tFLUSH_DIR\x10\t\x12\x0c\n\x08\x46LUSH_FP\x10\n\x12\x0b\n\x07JOIN_FP\x10\x0b\x12\x0c\n\x08\x46LUSH_FS\x10\x0c\x12\x0f\n\x0b\x46LUSH_FINAL\x10\r\x12\x07\n\x03\x45ND\x10\x0e\"<\n\x0cPauseRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x0f\n\rPauseResponse\"=\n\rResumeRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x10\n\x0eResumeResponse\"M\n\x0cLoginRequest\x12\x0f\n\x07\x61pi_key\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"&\n\rLoginResponse\x12\x15\n\ractive_entity\x18\x01 \x01(\t\"A\n\x11GetSummaryRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"?\n\x12GetSummaryResponse\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.SummaryItem\"=\n\rStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\")\n\x0eStatusResponse\x12\x17\n\x0frun_should_stop\x18\x01 \x01(\x08\"A\n\x11StopStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"-\n\x12StopStatusResponse\x12\x17\n\x0frun_should_stop\x18\x01 \x01(\x08\"D\n\x14NetworkStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"P\n\x15NetworkStatusResponse\x12\x37\n\x11network_responses\x18\x01 \x03(\x0b\x32\x1c.wandb_internal.HttpResponse\"D\n\x0cHttpResponse\x12\x18\n\x10http_status_code\x18\x01 \x01(\x05\x12\x1a\n\x12http_response_text\x18\x02 \x01(\t\"?\n\x0fPollExitRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\xbc\x01\n\x10PollExitResponse\x12\x0c\n\x04\x64one\x18\x01 \x01(\x08\x12\x32\n\x0b\x65xit_result\x18\x02 \x01(\x0b\x32\x1d.wandb_internal.RunExitResult\x12\x35\n\x0cpusher_stats\x18\x03 \x01(\x0b\x32\x1f.wandb_internal.FilePusherStats\x12/\n\x0b\x66ile_counts\x18\x04 \x01(\x0b\x32\x1a.wandb_internal.FileCounts\"\x13\n\x11SenderMarkRequest\"?\n\x11SenderReadRequest\x12\x14\n\x0cstart_offset\x18\x01 \x01(\x03\x12\x14\n\x0c\x66inal_offset\x18\x02 \x01(\x03\"m\n\x13StatusReportRequest\x12\x12\n\nrecord_num\x18\x01 \x01(\x03\x12\x13\n\x0bsent_offset\x18\x02 \x01(\x03\x12-\n\tsync_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"F\n\x14SummaryRecordRequest\x12.\n\x07summary\x18\x01 \x01(\x0b\x32\x1d.wandb_internal.SummaryRecord\"L\n\x16TelemetryRecordRequest\x12\x32\n\ttelemetry\x18\x01 \x01(\x0b\x32\x1f.wandb_internal.TelemetryRecord\"A\n\x11ServerInfoRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"|\n\x12ServerInfoResponse\x12-\n\nlocal_info\x18\x01 \x01(\x0b\x32\x19.wandb_internal.LocalInfo\x12\x37\n\x0fserver_messages\x18\x02 \x01(\x0b\x32\x1e.wandb_internal.ServerMessages\"=\n\x0eServerMessages\x12+\n\x04item\x18\x01 \x03(\x0b\x32\x1d.wandb_internal.ServerMessage\"e\n\rServerMessage\x12\x12\n\nplain_text\x18\x01 \x01(\t\x12\x10\n\x08utf_text\x18\x02 \x01(\t\x12\x11\n\thtml_text\x18\x03 \x01(\t\x12\x0c\n\x04type\x18\x04 \x01(\t\x12\r\n\x05level\x18\x05 \x01(\x05\"c\n\nFileCounts\x12\x13\n\x0bwandb_count\x18\x01 \x01(\x05\x12\x13\n\x0bmedia_count\x18\x02 \x01(\x05\x12\x16\n\x0e\x61rtifact_count\x18\x03 \x01(\x05\x12\x13\n\x0bother_count\x18\x04 \x01(\x05\"U\n\x0f\x46ilePusherStats\x12\x16\n\x0euploaded_bytes\x18\x01 \x01(\x03\x12\x13\n\x0btotal_bytes\x18\x02 \x01(\x03\x12\x15\n\rdeduped_bytes\x18\x03 \x01(\x03\"1\n\tLocalInfo\x12\x0f\n\x07version\x18\x01 \x01(\t\x12\x13\n\x0bout_of_date\x18\x02 \x01(\x08\"?\n\x0fShutdownRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x12\n\x10ShutdownResponse\"P\n\rAttachRequest\x12\x11\n\tattach_id\x18\x14 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"b\n\x0e\x41ttachResponse\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12(\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x19.wandb_internal.ErrorInfo\"\xd5\x02\n\x11TestInjectRequest\x12\x13\n\x0bhandler_exc\x18\x01 \x01(\x08\x12\x14\n\x0chandler_exit\x18\x02 \x01(\x08\x12\x15\n\rhandler_abort\x18\x03 \x01(\x08\x12\x12\n\nsender_exc\x18\x04 \x01(\x08\x12\x13\n\x0bsender_exit\x18\x05 \x01(\x08\x12\x14\n\x0csender_abort\x18\x06 \x01(\x08\x12\x0f\n\x07req_exc\x18\x07 \x01(\x08\x12\x10\n\x08req_exit\x18\x08 \x01(\x08\x12\x11\n\treq_abort\x18\t \x01(\x08\x12\x10\n\x08resp_exc\x18\n \x01(\x08\x12\x11\n\tresp_exit\x18\x0b \x01(\x08\x12\x12\n\nresp_abort\x18\x0c \x01(\x08\x12\x10\n\x08msg_drop\x18\r \x01(\x08\x12\x10\n\x08msg_hang\x18\x0e \x01(\x08\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x14\n\x12TestInjectResponse\"\x1e\n\rHistoryAction\x12\r\n\x05\x66lush\x18\x01 \x01(\x08\"\xca\x01\n\x15PartialHistoryRequest\x12)\n\x04item\x18\x01 \x03(\x0b\x32\x1b.wandb_internal.HistoryItem\x12)\n\x04step\x18\x02 \x01(\x0b\x32\x1b.wandb_internal.HistoryStep\x12-\n\x06\x61\x63tion\x18\x03 \x01(\x0b\x32\x1d.wandb_internal.HistoryAction\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x18\n\x16PartialHistoryResponse\"E\n\x15SampledHistoryRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"_\n\x12SampledHistoryItem\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x12\n\nnested_key\x18\x02 \x03(\t\x12\x14\n\x0cvalues_float\x18\x03 \x03(\x02\x12\x12\n\nvalues_int\x18\x04 \x03(\x03\"J\n\x16SampledHistoryResponse\x12\x30\n\x04item\x18\x01 \x03(\x0b\x32\".wandb_internal.SampledHistoryItem\"@\n\x10RunStatusRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"x\n\x11RunStatusResponse\x12\x18\n\x10sync_items_total\x18\x01 \x01(\x03\x12\x1a\n\x12sync_items_pending\x18\x02 \x01(\x03\x12-\n\tsync_time\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\"g\n\x0fRunStartRequest\x12&\n\x03run\x18\x01 \x01(\x0b\x32\x19.wandb_internal.RunRecord\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x12\n\x10RunStartResponse\"\\\n\x13\x43heckVersionRequest\x12\x17\n\x0f\x63urrent_version\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"]\n\x14\x43heckVersionResponse\x12\x17\n\x0fupgrade_message\x18\x01 \x01(\t\x12\x14\n\x0cyank_message\x18\x02 \x01(\t\x12\x16\n\x0e\x64\x65lete_message\x18\x03 \x01(\t\"\x8a\x01\n\x12LogArtifactRequest\x12\x30\n\x08\x61rtifact\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.ArtifactRecord\x12\x14\n\x0chistory_step\x18\x02 \x01(\x03\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"A\n\x13LogArtifactResponse\x12\x13\n\x0b\x61rtifact_id\x18\x01 \x01(\t\x12\x15\n\rerror_message\x18\x02 \x01(\t\"u\n\x13\x41rtifactSendRequest\x12\x30\n\x08\x61rtifact\x18\x01 \x01(\x0b\x32\x1e.wandb_internal.ArtifactRecord\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"#\n\x14\x41rtifactSendResponse\x12\x0b\n\x03xid\x18\x01 \x01(\t\"P\n\x13\x41rtifactPollRequest\x12\x0b\n\x03xid\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"Q\n\x14\x41rtifactPollResponse\x12\x13\n\x0b\x61rtifact_id\x18\x01 \x01(\t\x12\x15\n\rerror_message\x18\x02 \x01(\t\x12\r\n\x05ready\x18\x10 \x01(\x08\"N\n\x13\x41rtifactDoneRequest\x12\x13\n\x0b\x61rtifact_id\x18\x01 \x01(\t\x12\x15\n\rerror_message\x18\x02 \x01(\t\x12\x0b\n\x03xid\x18\x10 \x01(\t\"@\n\x10KeepaliveRequest\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x13\n\x11KeepaliveResponse\"h\n\x11UseArtifactRecord\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x0c\n\x04name\x18\x03 \x01(\t\x12+\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1b.wandb_internal._RecordInfo\"\x13\n\x11UseArtifactResult\"R\n\rCancelRequest\x12\x13\n\x0b\x63\x61ncel_slot\x18\x01 \x01(\t\x12,\n\x05_info\x18\xc8\x01 \x01(\x0b\x32\x1c.wandb_internal._RequestInfo\"\x10\n\x0e\x43\x61ncelResponseb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'wandb.proto.wandb_internal_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _RECORD._serialized_start=151
  _RECORD._serialized_end=1331
  _CONTROL._serialized_start=1334
  _CONTROL._serialized_end=1479
  _RESULT._serialized_start=1482
  _RESULT._serialized_end=1981
  _FINALRECORD._serialized_start=1983
  _FINALRECORD._serialized_end=2041
  _HEADERRECORD._serialized_start=2043
  _HEADERRECORD._serialized_end=2102
  _FOOTERRECORD._serialized_start=2104
  _FOOTERRECORD._serialized_end=2163
  _RUNRECORD._serialized_start=2166
  _RUNRECORD._serialized_end=2774
  _GITREPORECORD._serialized_start=2776
  _GITREPORECORD._serialized_end=2827
  _RUNUPDATERESULT._serialized_start=2829
  _RUNUPDATERESULT._serialized_end=2928
  _ERRORINFO._serialized_start=2931
  _ERRORINFO._serialized_end=3103
  _ERRORINFO_ERRORCODE._serialized_start=3012
  _ERRORINFO_ERRORCODE._serialized_end=3103
  _RUNEXITRECORD._serialized_start=3105
  _RUNEXITRECORD._serialized_end=3201
  _RUNEXITRESULT._serialized_start=3203
  _RUNEXITRESULT._serialized_end=3218
  _RUNPREEMPTINGRECORD._serialized_start=3220
  _RUNPREEMPTINGRECORD._serialized_end=3286
  _RUNPREEMPTINGRESULT._serialized_start=3288
  _RUNPREEMPTINGRESULT._serialized_end=3309
  _SETTINGSRECORD._serialized_start=3311
  _SETTINGSRECORD._serialized_end=3416
  _SETTINGSITEM._serialized_start=3418
  _SETTINGSITEM._serialized_end=3465
  _HISTORYSTEP._serialized_start=3467
  _HISTORYSTEP._serialized_end=3493
  _HISTORYRECORD._serialized_start=3496
  _HISTORYRECORD._serialized_end=3642
  _HISTORYITEM._serialized_start=3644
  _HISTORYITEM._serialized_end=3710
  _HISTORYRESULT._serialized_start=3712
  _HISTORYRESULT._serialized_end=3727
  _OUTPUTRECORD._serialized_start=3730
  _OUTPUTRECORD._serialized_end=3950
  _OUTPUTRECORD_OUTPUTTYPE._serialized_start=3914
  _OUTPUTRECORD_OUTPUTTYPE._serialized_end=3950
  _OUTPUTRESULT._serialized_start=3952
  _OUTPUTRESULT._serialized_end=3966
  _OUTPUTRAWRECORD._serialized_start=3969
  _OUTPUTRAWRECORD._serialized_end=4195
  _OUTPUTRAWRECORD_OUTPUTTYPE._serialized_start=3914
  _OUTPUTRAWRECORD_OUTPUTTYPE._serialized_end=3950
  _OUTPUTRAWRESULT._serialized_start=4197
  _OUTPUTRAWRESULT._serialized_end=4214
  _METRICRECORD._serialized_start=4217
  _METRICRECORD._serialized_end=4625
  _METRICRECORD_METRICGOAL._serialized_start=4559
  _METRICRECORD_METRICGOAL._serialized_end=4625
  _METRICRESULT._serialized_start=4627
  _METRICRESULT._serialized_end=4641
  _METRICOPTIONS._serialized_start=4643
  _METRICOPTIONS._serialized_end=4710
  _METRICCONTROL._serialized_start=4712
  _METRICCONTROL._serialized_end=4746
  _METRICSUMMARY._serialized_start=4748
  _METRICSUMMARY._serialized_end=4859
  _CONFIGRECORD._serialized_start=4862
  _CONFIGRECORD._serialized_end=5009
  _CONFIGITEM._serialized_start=5011
  _CONFIGITEM._serialized_end=5076
  _CONFIGRESULT._serialized_start=5078
  _CONFIGRESULT._serialized_end=5092
  _SUMMARYRECORD._serialized_start=5095
  _SUMMARYRECORD._serialized_end=5245
  _SUMMARYITEM._serialized_start=5247
  _SUMMARYITEM._serialized_end=5313
  _SUMMARYRESULT._serialized_start=5315
  _SUMMARYRESULT._serialized_end=5330
  _FILESRECORD._serialized_start=5332
  _FILESRECORD._serialized_end=5432
  _FILESITEM._serialized_start=5435
  _FILESITEM._serialized_end=5579
  _FILESITEM_POLICYTYPE._serialized_start=5539
  _FILESITEM_POLICYTYPE._serialized_end=5579
  _FILESRESULT._serialized_start=5581
  _FILESRESULT._serialized_end=5594
  _STATSRECORD._serialized_start=5597
  _STATSRECORD._serialized_end=5827
  _STATSRECORD_STATSTYPE._serialized_start=5804
  _STATSRECORD_STATSTYPE._serialized_end=5827
  _STATSITEM._serialized_start=5829
  _STATSITEM._serialized_end=5873
  _ARTIFACTRECORD._serialized_start=5876
  _ARTIFACTRECORD._serialized_end=6302
  _ARTIFACTMANIFEST._serialized_start=6305
  _ARTIFACTMANIFEST._serialized_end=6493
  _ARTIFACTMANIFESTENTRY._serialized_start=6496
  _ARTIFACTMANIFESTENTRY._serialized_end=6683
  _EXTRAITEM._serialized_start=6685
  _EXTRAITEM._serialized_end=6729
  _STORAGEPOLICYCONFIGITEM._serialized_start=6731
  _STORAGEPOLICYCONFIGITEM._serialized_end=6789
  _ARTIFACTRESULT._serialized_start=6791
  _ARTIFACTRESULT._serialized_end=6807
  _LINKARTIFACTRESULT._serialized_start=6809
  _LINKARTIFACTRESULT._serialized_end=6829
  _LINKARTIFACTRECORD._serialized_start=6832
  _LINKARTIFACTRECORD._serialized_end=7039
  _TBRECORD._serialized_start=7041
  _TBRECORD._serialized_end=7145
  _TBRESULT._serialized_start=7147
  _TBRESULT._serialized_end=7157
  _ALERTRECORD._serialized_start=7159
  _ALERTRECORD._serialized_end=7284
  _ALERTRESULT._serialized_start=7286
  _ALERTRESULT._serialized_end=7299
  _REQUEST._serialized_start=7302
  _REQUEST._serialized_end=8983
  _RESPONSE._serialized_start=8986
  _RESPONSE._serialized_end=10278
  _DEFERREQUEST._serialized_start=10281
  _DEFERREQUEST._serialized_end=10601
  _DEFERREQUEST_DEFERSTATE._serialized_start=10354
  _DEFERREQUEST_DEFERSTATE._serialized_end=10601
  _PAUSEREQUEST._serialized_start=10603
  _PAUSEREQUEST._serialized_end=10663
  _PAUSERESPONSE._serialized_start=10665
  _PAUSERESPONSE._serialized_end=10680
  _RESUMEREQUEST._serialized_start=10682
  _RESUMEREQUEST._serialized_end=10743
  _RESUMERESPONSE._serialized_start=10745
  _RESUMERESPONSE._serialized_end=10761
  _LOGINREQUEST._serialized_start=10763
  _LOGINREQUEST._serialized_end=10840
  _LOGINRESPONSE._serialized_start=10842
  _LOGINRESPONSE._serialized_end=10880
  _GETSUMMARYREQUEST._serialized_start=10882
  _GETSUMMARYREQUEST._serialized_end=10947
  _GETSUMMARYRESPONSE._serialized_start=10949
  _GETSUMMARYRESPONSE._serialized_end=11012
  _STATUSREQUEST._serialized_start=11014
  _STATUSREQUEST._serialized_end=11075
  _STATUSRESPONSE._serialized_start=11077
  _STATUSRESPONSE._serialized_end=11118
  _STOPSTATUSREQUEST._serialized_start=11120
  _STOPSTATUSREQUEST._serialized_end=11185
  _STOPSTATUSRESPONSE._serialized_start=11187
  _STOPSTATUSRESPONSE._serialized_end=11232
  _NETWORKSTATUSREQUEST._serialized_start=11234
  _NETWORKSTATUSREQUEST._serialized_end=11302
  _NETWORKSTATUSRESPONSE._serialized_start=11304
  _NETWORKSTATUSRESPONSE._serialized_end=11384
  _HTTPRESPONSE._serialized_start=11386
  _HTTPRESPONSE._serialized_end=11454
  _POLLEXITREQUEST._serialized_start=11456
  _POLLEXITREQUEST._serialized_end=11519
  _POLLEXITRESPONSE._serialized_start=11522
  _POLLEXITRESPONSE._serialized_end=11710
  _SENDERMARKREQUEST._serialized_start=11712
  _SENDERMARKREQUEST._serialized_end=11731
  _SENDERREADREQUEST._serialized_start=11733
  _SENDERREADREQUEST._serialized_end=11796
  _STATUSREPORTREQUEST._serialized_start=11798
  _STATUSREPORTREQUEST._serialized_end=11907
  _SUMMARYRECORDREQUEST._serialized_start=11909
  _SUMMARYRECORDREQUEST._serialized_end=11979
  _TELEMETRYRECORDREQUEST._serialized_start=11981
  _TELEMETRYRECORDREQUEST._serialized_end=12057
  _SERVERINFOREQUEST._serialized_start=12059
  _SERVERINFOREQUEST._serialized_end=12124
  _SERVERINFORESPONSE._serialized_start=12126
  _SERVERINFORESPONSE._serialized_end=12250
  _SERVERMESSAGES._serialized_start=12252
  _SERVERMESSAGES._serialized_end=12313
  _SERVERMESSAGE._serialized_start=12315
  _SERVERMESSAGE._serialized_end=12416
  _FILECOUNTS._serialized_start=12418
  _FILECOUNTS._serialized_end=12517
  _FILEPUSHERSTATS._serialized_start=12519
  _FILEPUSHERSTATS._serialized_end=12604
  _LOCALINFO._serialized_start=12606
  _LOCALINFO._serialized_end=12655
  _SHUTDOWNREQUEST._serialized_start=12657
  _SHUTDOWNREQUEST._serialized_end=12720
  _SHUTDOWNRESPONSE._serialized_start=12722
  _SHUTDOWNRESPONSE._serialized_end=12740
  _ATTACHREQUEST._serialized_start=12742
  _ATTACHREQUEST._serialized_end=12822
  _ATTACHRESPONSE._serialized_start=12824
  _ATTACHRESPONSE._serialized_end=12922
  _TESTINJECTREQUEST._serialized_start=12925
  _TESTINJECTREQUEST._serialized_end=13266
  _TESTINJECTRESPONSE._serialized_start=13268
  _TESTINJECTRESPONSE._serialized_end=13288
  _HISTORYACTION._serialized_start=13290
  _HISTORYACTION._serialized_end=13320
  _PARTIALHISTORYREQUEST._serialized_start=13323
  _PARTIALHISTORYREQUEST._serialized_end=13525
  _PARTIALHISTORYRESPONSE._serialized_start=13527
  _PARTIALHISTORYRESPONSE._serialized_end=13551
  _SAMPLEDHISTORYREQUEST._serialized_start=13553
  _SAMPLEDHISTORYREQUEST._serialized_end=13622
  _SAMPLEDHISTORYITEM._serialized_start=13624
  _SAMPLEDHISTORYITEM._serialized_end=13719
  _SAMPLEDHISTORYRESPONSE._serialized_start=13721
  _SAMPLEDHISTORYRESPONSE._serialized_end=13795
  _RUNSTATUSREQUEST._serialized_start=13797
  _RUNSTATUSREQUEST._serialized_end=13861
  _RUNSTATUSRESPONSE._serialized_start=13863
  _RUNSTATUSRESPONSE._serialized_end=13983
  _RUNSTARTREQUEST._serialized_start=13985
  _RUNSTARTREQUEST._serialized_end=14088
  _RUNSTARTRESPONSE._serialized_start=14090
  _RUNSTARTRESPONSE._serialized_end=14108
  _CHECKVERSIONREQUEST._serialized_start=14110
  _CHECKVERSIONREQUEST._serialized_end=14202
  _CHECKVERSIONRESPONSE._serialized_start=14204
  _CHECKVERSIONRESPONSE._serialized_end=14297
  _LOGARTIFACTREQUEST._serialized_start=14300
  _LOGARTIFACTREQUEST._serialized_end=14438
  _LOGARTIFACTRESPONSE._serialized_start=14440
  _LOGARTIFACTRESPONSE._serialized_end=14505
  _ARTIFACTSENDREQUEST._serialized_start=14507
  _ARTIFACTSENDREQUEST._serialized_end=14624
  _ARTIFACTSENDRESPONSE._serialized_start=14626
  _ARTIFACTSENDRESPONSE._serialized_end=14661
  _ARTIFACTPOLLREQUEST._serialized_start=14663
  _ARTIFACTPOLLREQUEST._serialized_end=14743
  _ARTIFACTPOLLRESPONSE._serialized_start=14745
  _ARTIFACTPOLLRESPONSE._serialized_end=14826
  _ARTIFACTDONEREQUEST._serialized_start=14828
  _ARTIFACTDONEREQUEST._serialized_end=14906
  _KEEPALIVEREQUEST._serialized_start=14908
  _KEEPALIVEREQUEST._serialized_end=14972
  _KEEPALIVERESPONSE._serialized_start=14974
  _KEEPALIVERESPONSE._serialized_end=14993
  _USEARTIFACTRECORD._serialized_start=14995
  _USEARTIFACTRECORD._serialized_end=15099
  _USEARTIFACTRESULT._serialized_start=15101
  _USEARTIFACTRESULT._serialized_end=15120
  _CANCELREQUEST._serialized_start=15122
  _CANCELREQUEST._serialized_end=15204
  _CANCELRESPONSE._serialized_start=15206
  _CANCELRESPONSE._serialized_end=15222
# @@protoc_insertion_point(module_scope)
