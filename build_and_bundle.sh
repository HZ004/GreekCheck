#!/bin/bash
set -e

# 1. Go to frontend folder, install deps and build React app
cd frontend
npm install
npm run build

# 2. Copy the React build folder to the Python component folder
cd ..
rm -rf greek_component/frontend/build
mkdir -p greek_component/frontend
cp -r frontend/build greek_component/frontend/
