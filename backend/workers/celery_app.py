"""Celery entrypoint. Start with: celery -A backend.workers.celery_app.celery_app worker --loglevel=INFO"""
import os
from celery import Celery
broker=os.getenv("CELERY_BROKER_URL","redis://redis:6379/0")
celery_app=Celery("deallens",broker=broker,backend=os.getenv("CELERY_RESULT_BACKEND",broker))
celery_app.conf.update(task_track_started=True,task_acks_late=True,
                       task_reject_on_worker_lost=True,
                       task_default_retry_delay=5,task_max_retries=3,
                       broker_connection_retry_on_startup=True)
