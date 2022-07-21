import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
from subprocess import run

from conda.common.compat import on_mac
from conda.exceptions import CondaExitZero
from conda.gateways.connection.adapters.s3 import S3Adapter
from conda.testing.integration import _get_temp_prefix, run_command
import pytest

from .channel_testing_utils import (
    create_with_channel,
    create_with_channel_in_process,
    http_server_auth_basic,
    http_server_auth_basic_email,
    http_server_auth_none,
    http_server_auth_token,
    s3_server,
)


def test_http_server_auth_none(http_server_auth_none):
    create_with_channel(http_server_auth_none)


def test_http_server_auth_basic(http_server_auth_basic):
    create_with_channel(http_server_auth_basic)


def test_http_server_auth_basic_email(http_server_auth_basic_email):
    create_with_channel(http_server_auth_basic_email)


def test_http_server_auth_token(http_server_auth_token):
    create_with_channel(http_server_auth_token)


def prepare_s3_server(endpoint, bucket_name):
    # prepare the s3 connection for our minio instance
    import boto3
    from botocore.client import Config

    # Make the minio bucket public first
    # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-example-bucket-policies.html#set-a-bucket-policy
    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    bucket_policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AddPerm",
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": ["s3:GetObject"],
                    "Resource": f"arn:aws:s3:::{bucket_name}/*",
                }
            ],
        }
    )
    client.put_bucket_policy(Bucket=bucket_name, Policy=bucket_policy)

    # Minio has to start with an empty directory; once available,
    # we can import all channel files by "uploading" them
    mamba_repo_dir = Path(__file__).parent / "data" / "mamba_repo"
    for path in (mamba_repo_dir / "noarch").iterdir():
        key = path.relative_to(mamba_repo_dir)
        client.upload_file(str(path), bucket_name, str(key), ExtraArgs={"ACL": "public-read"})


def _s3_adapter_send_boto3_patch_factory(endpoint):
    def _send_boto3(self, boto3, resp, request):
        """
        We use this to patch S3Adapter._send_boto3 function so
        it connects to our local instance. All we are changing here
        is the call to `session.resource(...)`.
        """
        from botocore.exceptions import BotoCoreError, ClientError
        from botocore.client import Config
        from conda.common.compat import ensure_binary
        from conda.common.url import url_to_s3_info
        from conda.gateways.connection import CaseInsensitiveDict

        bucket_name, key_string = url_to_s3_info(request.url)
        # https://github.com/conda/conda/issues/8993
        # creating a separate boto3 session to make this thread safe
        session = boto3.session.Session()
        # create a resource client using this thread's session object
        s3 = session.resource(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        # finally get the S3 object
        key = s3.Object(bucket_name, key_string[1:])

        try:
            response = key.get()
        except (BotoCoreError, ClientError) as e:
            resp.status_code = 404
            message = {
                "error": "error downloading file from s3",
                "path": request.url,
                "exception": repr(e),
            }
            resp.raw = self._write_tempfile(lambda x: x.write(ensure_binary(json.dumps(message))))
            resp.close = resp.raw.close
            return resp

        key_headers = response["ResponseMetadata"]["HTTPHeaders"]
        resp.headers = CaseInsensitiveDict(
            {
                "Content-Type": key_headers.get("content-type", "text/plain"),
                "Content-Length": key_headers["content-length"],
                "Last-Modified": key_headers["last-modified"],
            }
        )

        resp.raw = self._write_tempfile(key.download_fileobj)
        resp.close = resp.raw.close

        return resp

    return _send_boto3


@pytest.mark.skipif(
    run(["minio", "-v"], check=False).returncode != 0,
    reason="Minio server not available on PATH",
)
def test_s3_server(s3_server):
    endpoint, bucket_name = s3_server.rsplit("/", 1)
    prepare_s3_server(endpoint, bucket_name)

    with pytest.raises(CondaExitZero), patch.object(
        S3Adapter, "_send_boto3", _s3_adapter_send_boto3_patch_factory(endpoint)
    ):
        create_with_channel_in_process(f"s3://{bucket_name}", no_capture=True)


def test_channel_matchspec():
    stdout, stderr, _ = run_command(
        "create",
        _get_temp_prefix(),
        "--experimental-solver=libmamba",
        "--json",
        "--override-channels",
        "-c",
        "defaults",
        "conda-forge::libblas=*=*openblas",
        "python=3.9",
    )
    result = json.loads(stdout)
    assert result["success"] is True
    for record in result["actions"]["LINK"]:
        if record["name"] == "numpy":
            assert record["channel"] == "conda-forge"
        elif record["name"] == "python":
            assert record["channel"] == "pkgs/main"
