# Product Overview

This document describes the architecture of the recommendation engine.

## Data Ingestion

The service ingests events from Kafka topics every minute.
Each event contains user_id, session metadata, and interaction context.

## Ranking Model

The ranking model is trained weekly with feature drift checks.
We use cross-validation and monitor NDCG and MRR metrics.

## Deployment

The model is deployed as a stateless container in Kubernetes.
Canary rollout is used before full production deployment.

