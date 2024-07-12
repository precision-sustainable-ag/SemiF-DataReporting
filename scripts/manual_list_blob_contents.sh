#!/usr/bin/env bash

KEYSPATH=$1
BLOB_CONTAINER=$2
LEVELS=$3
OUTPUTDIR=$4

URL=$(cat $KEYSPATH | shyaml get-value blobs.$BLOB_CONTAINER.url)
SAS=$(cat $KEYSPATH | shyaml get-value blobs.$BLOB_CONTAINER.sas_token)

azcopy ls $URL$SAS | cut -d/ -f 1-$LEVELS | awk '!a[$0]++' > $OUTPUTDIR/$BLOB_CONTAINER.txt
