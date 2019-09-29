I implemented the XOR tag scheme for sending data though the network. The following is an explanation of the output:

The data being sent is the rank, but only for demonstration purposes, any data can be sent.

0 --> 1 means PE 0 is sending to PE 1

0 -> 8 (=) -> 12 means SW 8 received from 0 and sent to 12 using the straight configuration
1 -> 9 (X) -> 15 means SW 9 received from 1 and sent to 15 using the crossed configuration

1 got 0 means process 1 received data with value 0, for the demonstration this means the data was sent from 0.

Thus we can see that 0 sent to 1 and 1 got data from zero.
