1.Before starting the experiment, use nvitop to check for idle GPUs.

2.Then use the command lsof -i:<port_number> to check if a port is available.
  For example, lsof -i:8080 checks whether port 8080 is in use. If the command returns no output, it means the port is free and available. If there is output, it means the port is already in use.
  Commonly used ports are in the range of 1024–49151. I've used 8080, 9090, 8000, and 5000—all of which work.
  The port number currently needs to be modified directly in the code, as it is not yet passed as an argument.

3.After starting the experiment, remember to record the PIDs of the four threads: serve, client_1, client_2, and client_3.

4.The output of each thread can be found in its corresponding .out file.
  After starting the three clients, do not close VS Code immediately. Wait until one full round has completed, then check each .out file for any errors. If there are no issues, you can safely close VS Code.

5.There are two ways to check for background threads:
  a. If you're in the same terminal, use jobs -l.
  b. If you've closed and reopened VS Code, use ps -def.
  To forcefully terminate a thread, use kill -9 <pid>, for example:
  kill -9 2809860
