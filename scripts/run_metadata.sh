source settings/config.sh

METADATA_PORT=$GLOBAL_METADATA_PORT

echo "Run metadata"
echo ""
sleep 1
./build/bin/metadata_storage 0.0.0.0 $METADATA_PORT