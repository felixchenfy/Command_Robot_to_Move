
# 1. Server

Run: **$ python2 voice_listener.py**  
It waits for the speech command from "voice_speaker.py"

This script will be imported by a ROS script for inputting the speech command of user.

# 2. Client
Run: **$ python3 voice_speaker.py --device 0**  

Press "R" to record audio -- Your voice will be recorded by your laptop's microphone.  
Then, this script classifies your speech command, and saves the result to a txt file (See below). 


# 3. Communication

The server and the client are communicating through a txt file, "../../comm/voice_comm.txt", which is defined by the "../../config/config.py".

# Others
For more details about speech commands classification, please see my project:  
https://github.com/felixchenfy/Speech-Commands-Classification-by-LSTM-PyTorch.git