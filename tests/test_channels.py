import json
from pathlib import Path
from unittest.mock import patch
from subprocess import run

from conda.exceptions import CondaExitZero
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

try:
    run(["minio", "-v"], check=True)
    have_minio = True
except Exception:
    have_minio = False


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


@pytest.mark.skipif(not have_minio, reason="Minio server not available")
def test_s3_server(s3_server):
    import boto3
    from botocore.client import Config

    endpoint, bucket_name = s3_server.rsplit("/", 1)
    prepare_s3_server(endpoint, bucket_name)

    # We patch the default kwargs values in boto3.session.Session.resource(...)
    # which is used in conda.gateways.connection.s3.S3Adapter to initialize the S3
    # connection; otherwise it would default to a real AWS instance
    patched_defaults = (
        "us-east-1",  # region_name
        None,  # api_version
        True,  # use_ssl
        None,  # verify
        endpoint,  # endpoint_url
        "minioadmin",  # aws_access_key_id
        "minioadmin",  # aws_secret_access_key
        None,  # aws_session_token
        Config(signature_version="s3v4"),  # config
    )
    with pytest.raises(CondaExitZero), patch.object(
        boto3.session.Session.resource, "__defaults__", patched_defaults
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
