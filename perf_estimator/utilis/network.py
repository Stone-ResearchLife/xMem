import ipaddress


def verify_ip_in_subnet(ip, subnet) -> bool:
    """Verify if an ip address is in a subnet.

    Args:
        ip (str): an ip address
        subnet (str): a subnet

    Returns:
        bool: True if the ip address is in the subnet, False otherwise
    """
    ip = ipaddress.ip_address(ip)
    subnet = ipaddress.ip_network(subnet)
    return ip in subnet


def generate_ip(subnet, last_index):
    """Generate an ip address based on the subnet and the last index.

    Args:
        subnet (str): a subnet
        last_index (int): the last index of the ip address.

    Examples:
        generate_ip("192.168.0.0/24", 100)
        # Output: 192.168.0.101

    Returns:
        str: a new ip address

    """
    network = ipaddress.ip_network(subnet)
    new_ip = network.network_address + last_index
    return str(new_ip)
