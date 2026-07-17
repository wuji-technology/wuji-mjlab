#!/usr/bin/env bash
# feishu-sign.sh
#
# Compute the HMAC-SHA256 signature required by Feishu custom-bot webhooks
# that have signature verification enabled.
#
# Feishu signing algorithm:
#   1. string_to_sign = "<timestamp>\n<secret>"
#   2. signature = base64(hmac_sha256(key=string_to_sign, data=""))
#
# Usage:
#   feishu-sign.sh <timestamp> <secret>
#
# Output (stdout):
#   <base64-signature>

set -euo pipefail

TIMESTAMP="${1:?timestamp required}"
SECRET="${2:?secret required}"

STRING_TO_SIGN="${TIMESTAMP}
${SECRET}"

printf '' | openssl dgst -sha256 -hmac "$STRING_TO_SIGN" -binary | base64
