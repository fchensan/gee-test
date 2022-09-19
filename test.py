import numpy as np
from nfstream import NFStreamer
from statistics import stdev
from math import log2
from collections import Counter
import torch
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class VAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def training_step(self, batch, batch_idx):
        x = batch['feature']
        recon_x, mu, logvar = self(x)
        loss = self.loss_function(recon_x, x, mu, logvar)

        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)


class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # layer 1
            nn.Linear(
                in_features=69,
                out_features=512
            ),
            nn.ReLU(),
            # layer 2
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.ReLU(),
            # layer 3
            nn.Linear(
                in_features=512,
                out_features=1024
            ),
            nn.ReLU(),
        )

        # output
        self.mu = nn.Linear(
            in_features=1024,
            out_features=100
        )
        self.logvar = nn.Linear(
            in_features=1024,
            out_features=100
        )

    def forward(self, x):
        h = self.fc(x)
        return self.mu(h), self.logvar(h)


class Decoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            # layer 1
            nn.Linear(
                in_features=100,
                out_features=1024
            ),
            nn.ReLU(),
            # layer 2
            nn.Linear(
                in_features=1024,
                out_features=512
            ),
            nn.ReLU(),
            # layer 3
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.ReLU(),
            # output
            nn.Linear(
                in_features=512,
                out_features=69
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)

model_path = "../Downloads/model/vae.model"
model = VAE.load_from_checkpoint(checkpoint_path=model_path)

my_streamer = NFStreamer(source="enp0s6",# Live capture mode. 
                         # Disable L7 dissection for readability purpose only.
                         n_dissections=0,
                         system_visibility_poll_ms=100,
                         system_visibility_mode=1,
                         active_timeout=3,
                         performance_report=10)

def can_run_classification(flows):
    if len(flows) > 3: 
        return True
    return False

# Function from github.com/munhouiani/GEE
def entropy(data):
        """
        Extract shannon entropy of a given list
        :param grouped_data: grouped data
        :type grouped_data: pd.Series
        :return: entropy
        :rtype: float
        """

        ent = 0.0
        if len(data) <= 1:
            return ent

        counter = Counter(data)
        probs = [c / len(data) for c in counter.values()]

        for p in probs:
            if p > 0.0:
                ent -= p * log2(p)

        return ent

# Function from github.com/munhouiani/GEE
def port_proportion(port_list):
        """
        Extract port proportion of a given pandas Series
        :param grouped_data: grouped data
        :type grouped_data: pd.Series
        :return: standard deviation value
        :rtype: list of float
        """

        common_port = {
            20: 0,  # FTP
            21: 1,  # FTP
            22: 2,  # SSH
            23: 3,  # Telnet
            25: 4,  # SMTP
            50: 5,  # IPSec
            51: 6,  # IPSec
            53: 7,  # DNS
            67: 8,  # DHCP
            68: 9,  # DHCP
            69: 10,  # TFTP
            80: 11,  # HTTP
            110: 12,  # POP3
            119: 13,  # NNTP
            123: 14,  # NTP
            135: 15,  # RPC
            136: 16,  # NetBios
            137: 17,  # NetBios
            138: 18,  # NetBios
            139: 19,  # NetBios
            143: 20,  # IMAP
            161: 21,  # SNMP
            162: 22,  # SNMP
            389: 23,  # LDAP
            443: 24,  # HTTPS
            3389: 25,  # RDP
        }

        proportion = [0.0] * (len(common_port) + 1)
        for port in port_list:
            idx = common_port.get(port)
            if idx is None:
                idx = -1
            proportion[idx] += 1

        proportion = [x / sum(proportion) for x in proportion]

        return proportion

def aggregate_features(flows):
    features = []
    
    durations = [flow["duration"] for flow in flows]
    mean_duration = sum(durations) / len(flows)
    std_duration = stdev(durations) 

    num_of_packets = [flow["num_of_packets"] for flow in flows]
    mean_num_of_packets = sum(num_of_packets) / len(flows)
    std_num_of_packets = stdev(num_of_packets)

    num_of_bytes = [flow["num_of_bytes"] for flow in flows]
    mean_num_of_bytes = sum(num_of_bytes) / len(flows)
    std_num_of_bytes = stdev(num_of_bytes)

    packet_rates = [flow["packet_rate"] for flow in flows]
    mean_packet_rate = sum(packet_rates) / len(flows)
    std_mean_packet_rate = stdev(packet_rates)
 
    byte_rates = [flow["byte_rate"] for flow in flows]
    mean_byte_rate = sum(byte_rates) / len(flows)
    std_byte_rate = stdev(byte_rates)

    protocols = [flow["protocol"] for flow in flows]
    entropy_protocol = entropy(protocols) 

    dst_ips = [flow["dst_ip"] for flow in flows]
    entropy_dst_ip = entropy(dst_ips)

    src_ports = [flow["src_port"] for flow in flows]
    entropy_src_port = entropy(src_ports)
    
    dst_ports = [flow["dst_port"] for flow in flows]
    entropy_dst_port = entropy(dst_ports)

    entropy_flags = 0.1

    proportion_src_port = port_proportion(src_ports)
    proportion_dst_port = port_proportion(dst_ports)

    return [mean_duration, mean_num_of_packets, mean_num_of_bytes, mean_packet_rate, mean_byte_rate,
        std_duration, std_num_of_packets, std_num_of_bytes, std_mean_packet_rate, std_byte_rate,
        entropy_protocol, entropy_dst_ip, entropy_src_port, entropy_dst_port,
        entropy_flags] + proportion_src_port + proportion_dst_port

def classify(features):
    features = np.nan_to_num(features, nan=0)
    input_as_tensor = torch.as_tensor(features, dtype=torch.float32)
    reconstruction = model(input_as_tensor)[0].detach().numpy()
    print(np.square(reconstruction - features).mean())

flows_per_ip_src = {}

flows_aggregations_to_normalize = []
normalization = []

for flow in my_streamer:
    flow_data = {}

    flow_data["duration"] = flow.bidirectional_duration_ms
    if flow_data["duration"] == 0:
        continue
    flow_data["num_of_packets"] = flow.bidirectional_packets
    flow_data["num_of_bytes"] = flow.bidirectional_bytes
    flow_data["packet_rate"] = flow.bidirectional_packets / flow_data["duration"] 
    flow_data["byte_rate"] = flow.bidirectional_bytes / flow_data["duration"]
    flow_data["protocol"] = flow.protocol
    flow_data["dst_ip"] = flow.dst_ip
    flow_data["src_port"] = flow.src_port
    flow_data["dst_port"] = flow.dst_port
    flow_data["flags"] = 0 
    #flow_data["flags"] = [flow.syn,
    #        flow.cwr,
    #        flow.ece,
    #        flow.urg,
    #        flow.ack,
    #        flow.psh,
    #        flow.rst,
    #        flow.fin]

    if flow.src_ip in flows_per_ip_src:
        flows_per_ip_src[flow.src_ip] += [flow_data]
    else:
        flows_per_ip_src[flow.src_ip] = [flow_data]
    
    if can_run_classification(flows_per_ip_src[flow.src_ip]):
        aggregation = aggregate_features(flows_per_ip_src[flow.src_ip])
        if normalization == []:
            flows_aggregations_to_normalize += [aggregation]
            if len(flows_aggregations_to_normalize) == 3:
                normalization = np.mean(flows_aggregations_to_normalize, axis=0) 
                # np.nan_to_num(normalization, copy=False, nan=0)
        else:
            aggregation_normalized = np.divide(aggregation, normalization).tolist()
            classify(aggregation_normalized[:13]+aggregation[13:])
        flows_per_ip_src[flow.src_ip] = []

    print("new flow!")
