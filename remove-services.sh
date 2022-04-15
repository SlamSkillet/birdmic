#!/bin/bash

# Stop services
find /etc/systemd/system -type f -printf '%P\n' | grep birdaudio- | grep .service | sudo xargs systemctl stop
find /etc/systemd/system -type f -printf '%P\n' | grep birdaudio- | grep .timer | sudo xargs systemctl stop

# Remove symlinks
find /etc/systemd/system | grep birdaudio- | grep .service | sudo xargs systemctl disable --now
find /etc/systemd/system | grep birdaudio- | grep .timer | sudo xargs systemctl disable --now

# Delete files
find /etc/systemd/system | grep birdaudio- | xargs sudo rm -rf

# Restart systemd daemon
sudo systemctl daemon-reload