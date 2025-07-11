import socket
import numpy as np
import casperfpga
import argparse
import time

# ----- Argument Parsing -----
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="fpg file to upload to red-pitaya")
parser.add_argument("-r", "--redpitaya", help="Red-Pitaya hostname or  IP address")
parser.add_argument("-a", "--accums", help="Number of accumulations", default=4)
args = parser.parse_args()

# ----- FPGA Configuration -----
fpga = casperfpga.CasperFpga(args.redpitaya)
fpga.upload_to_ram_and_program(args.file)

fft_len = 256
acc_len = int(args.accums)
fpga.write_int('acc_len', acc_len)
fpga.write_int('snap_gap', 10)
fpga.write_int('reg_cntrl', 1)

# ----- UDP Configuration -----
UDP_IP = "255.255.255.255"  # Destination
UDP_PORT = 8052
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# ----- Main Loop -----
while True:
    start_time = time.time()

    # Trigger snapshot
    fpga.snapshots.accum1_snap_ss.arm()
    snap = fpga.snapshots.accum1_snap_ss.read(arm=False)['data']
    spectrum = np.array(snap['P_acc0'], dtype=np.int32)

    # Convert to bytes
    spectrum_bytes = spectrum.tobytes()

    # Send over UDP
    sock.sendto(spectrum_bytes, (UDP_IP, UDP_PORT))
    print(f"Sent {len(spectrum_bytes)} bytes over UDP.")

    end_time = time.time()
    print("Time per iteration: {:.5f} seconds".format(end_time - start_time))
    # time.sleep(0.028)  # Control rate of transmission
