#!/bin/bash

echo 'Removing files in /processed folder...'
echo $(ls -a ~/birdaudio/recordings/processed)
rm -rf ~/birdaudio/recordings/processed/*
echo 'Files removed.'