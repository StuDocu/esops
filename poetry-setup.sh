#!/bin/bash

export AWS_PROFILE=ci-user
export CODEARTIFACT_USER=aws
export CODEARTIFACT_AUTH_TOKEN=$(aws codeartifact get-authorization-token --domain studocu --domain-owner 179143349265 --region eu-west-1 --query authorizationToken --output text)

pip config set global.index-url https://aws:$CODEARTIFACT_AUTH_TOKEN@studocu-179143349265.d.codeartifact.eu-west-1.amazonaws.com/pypi/studocu-internal/simple/
pip config set global.extra-index-url https://aws:$CODEARTIFACT_AUTH_TOKEN@studocu-179143349265.d.codeartifact.eu-west-1.amazonaws.com/pypi/studocu-internal-release/simple/

# Check if poetry is installed
if ! command -v poetry &> /dev/null
then
    echo "Poetry not found, installing..."
    pip install poetry==1.6.1
else
    echo "Poetry is already installed."
fi

# Configure poetry
poetry config repositories.studocu-internal https://studocu-179143349265.d.codeartifact.eu-west-1.amazonaws.com/pypi/studocu-internal/
poetry config http-basic.studocu-internal aws "$CODEARTIFACT_AUTH_TOKEN"
poetry config repositories.studocu-internal-release https://studocu-179143349265.d.codeartifact.eu-west-1.amazonaws.com/pypi/studocu-internal-release/
poetry config http-basic.studocu-internal-release aws "$CODEARTIFACT_AUTH_TOKEN"
