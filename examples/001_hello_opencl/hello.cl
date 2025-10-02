__kernel void hello_kernel(__global char* message) {
    int gid = get_global_id(0);
    
    // Only the first work-item writes the message
    if (gid == 0) {
        message[0] = 'H';
        message[1] = 'e';
        message[2] = 'l';
        message[3] = 'l';
        message[4] = 'o';
        message[5] = ' ';
        message[6] = 'f';
        message[7] = 'r';
        message[8] = 'o';
        message[9] = 'm';
        message[10] = ' ';
        message[11] = 'G';
        message[12] = 'P';
        message[13] = 'U';
        message[14] = '!';
        message[15] = '\0';
    }
}