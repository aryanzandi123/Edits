from dotenv import load_dotenv

load_dotenv()

from runner import run_full_job
import threading

protein_name = "ATXN3"
jobs = {}
jobs[protein_name] = {"status": "processing", "progress": "Initializing pipeline..."}
thread = threading.Thread(target=run_full_job, args=(protein_name, jobs, threading.Lock()))
thread.daemon = True
thread.start()