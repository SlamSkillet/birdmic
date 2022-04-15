#!/bin/bash

# Copy services from repo to systemd directory
sudo cp ./services/* /etc/systemd/system
find /etc/systemd/system | grep birdaudio- | xargs sudo chmod +x

# Create symlinks
find /etc/systemd/system | grep birdaudio- | grep .service | sudo xargs systemctl enable
find /etc/systemd/system | grep birdaudio- | grep .timer | sudo xargs systemctl enable

# Start services
find /etc/systemd/system -type f -printf '%P\n' | grep birdaudio- | grep .service | sudo xargs systemctl start
find /etc/systemd/system -type f -printf '%P\n' | grep birdaudio- | grep .timer | sudo xargs systemctl start
