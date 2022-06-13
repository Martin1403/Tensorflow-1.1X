#!/usr/bin/env bash
python export.py
checkpoint/convert_graph --in_graph=graph/output_graph.pb --out_graph=graph/output_graph.pbmm
python export.py --export_tflite
cp checkpoint/output_graph.scorer graph/output_graph.scorer