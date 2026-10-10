This launcher is designed to submit each gene KOs with a single job so that the computational resources 
are efficiently utilized and each knockout experiment is independently managed.

Usage:
    python job/write_for_new_job.py
For each launch, a hash code will be generated, and the launcher command will be saved as well as 
the associated simulation configuration.

to-dos:
The current write_for_new_job also saves the submit command for each job, but the recent saves are
deleted due to concerns rasing about quota (file number restrictions).