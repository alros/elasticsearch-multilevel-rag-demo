#!/bin/sh

docker pull docker.elastic.co/elasticsearch/elasticsearch:8.11.0
docker pull ghcr.io/nlmatics/nlm-ingestor:latest
docker pull ollama/ollama

# install dependencies
pip install -r requirements.txt &

# start elasticsearch
docker run -p 9200:9200 -d --name elasticsearch \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "xpack.security.http.ssl.enabled=false" \
  -e "xpack.license.self_generated.type=trial" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0 &

# start nlm-ingestor
docker run -p 5010:5001 -d --name nlmingestor \
  ghcr.io/nlmatics/nlm-ingestor:latest &

# start ollama
brew install ollama
ollama serve &
ollama pull mistral
