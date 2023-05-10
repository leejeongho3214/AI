# --------------------------------
# Setup
# --------------------------------
export REPO_DIR=$PWD
if [ ! -d $REPO_DIR/models ] ; then
    mkdir -p $REPO_DIR/models
fi
BLOB='https://datarelease.blob.core.windows.net/metro'


# --------------------------------
# Download our pre-trained models
# --------------------------------
if [ ! -d $REPO_DIR/models/graphormer_release ] ; then
    mkdir -p $REPO_DIR/models/graphormer_release
fi
# (1) Mesh Graphormer for human mesh reconstruction (trained on H3.6M + COCO + MuCO + UP3D + MPII)

# (3) Mesh Graphormer for hand mesh reconstruction (trained on FreiHAND)
wget -nc $BLOB/models/graphormer_hand_state_dict.bin -O $REPO_DIR/models/graphormer_release/graphormer_hand_state_dict.bin


