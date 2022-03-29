#!/usr/bin/bash

# Variable $1 from command line is type of mode file: leakyvec, guidedvec etc


# Remote copy outputs folder from cluster to current folder 
scp -r $CLUSTER1:$COEUS_HOME/$1/outputs ./$1

