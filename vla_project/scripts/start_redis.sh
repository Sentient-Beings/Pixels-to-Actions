#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="$SCRIPT_DIR/../configs"
REDIS_CONF_PATH="$CONFIGS_DIR/redis.conf"

redis-server "$REDIS_CONF_PATH"