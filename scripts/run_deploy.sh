echo "Deploy models"
echo ""

sleep 1
FI_EFA_ENABLE_SHM_TRANSFER=0 python build/bin/deploy_file.py settings/storage_list.txt build/resource/model_distribution.txt build/resource/kv.bin deploy_file 4000000000