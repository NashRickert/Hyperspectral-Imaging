#!/bin/bash

# Exit on error
set -e

# Function for logging
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function for error handling
error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

# Function to validate the network adapter
validate_adapter() {
    local adapter=$1
    
    # Check if adapter exists
    if ! ip link show "$adapter" &>/dev/null; then
        error_exit "Network adapter '$adapter' does not exist. Please check available adapters with 'ip link show'"
    fi
    
    log "Network adapter '$adapter' validated successfully"
}

# Function to validate IP address format
validate_ip() {
    local ip=$1
    local valid_ip_regex='^([0-9]{1,3}\.){3}[0-9]{1,3}$'
    
    if ! [[ $ip =~ $valid_ip_regex ]]; then
        error_exit "Invalid IP address format: $ip"
    fi
    
    # Check each octet is between 0-255
    IFS='.' read -r -a octets <<< "$ip"
    for octet in "${octets[@]}"; do
        if [[ $octet -lt 0 || $octet -gt 255 ]]; then
            error_exit "Invalid IP address: $ip (each octet must be between 0-255)"
        fi
    done
    
    log "IP address '$ip' validated successfully"
}

# Function to configure static IP
configure_static_ip() {
    local adapter=$1
    local ip_address=$2
    
    log "Configuring static IP $ip_address for adapter $adapter..."
    
    # Bring down interface
    sudo ip link set "$adapter" down || error_exit "Failed to bring down interface $adapter"
    
    # Configure static IP (with /24 subnet mask)
    sudo ip addr flush dev "$adapter" || error_exit "Failed to flush IP addresses on $adapter"
    sudo ip addr add "$ip_address/24" dev "$adapter" || error_exit "Failed to set IP address $ip_address on $adapter"
    
    # Bring interface back up
    sudo ip link set "$adapter" up || error_exit "Failed to bring up interface $adapter"
    
    # Verify IP was set correctly
    if ! ip addr show "$adapter" | grep -q "$ip_address"; then
        error_exit "Failed to verify static IP configuration"
    fi
    
    log "Static IP $ip_address configured successfully on $adapter"
}

# Function to get IP address of adapter
get_ip_address() {
    local adapter=$1
    local ip_address
    
    ip_address=$(ip -4 addr show "$adapter" | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
    
    if [ -z "$ip_address" ]; then
        error_exit "Could not determine IP address for adapter '$adapter'"
    fi
    
    echo "$ip_address"
}

# Function to cleanup NFS and TFTP servers
cleanup_services() {
    local adapter=$1
    
    log "Starting cleanup process..."
    
    # Stop and disable NFS server
    log "Stopping NFS server..."
    sudo systemctl stop nfs-kernel-server &>/dev/null || true
    sudo systemctl disable nfs-kernel-server &>/dev/null || true
    
    # Stop and disable TFTP server
    log "Stopping TFTP server..."
    sudo systemctl stop tftpd-hpa &>/dev/null || true
    sudo systemctl disable tftpd-hpa &>/dev/null || true
    
    # Remove NFS exports
    log "Removing NFS exports..."
    sudo sed -i '/^\/srv\/nfs\/shared/d' /etc/exports
    sudo exportfs -ra &>/dev/null || true
    
    # Remove shared directories
    log "Removing shared directories..."
    sudo rm -rf /srv/nfs/shared
    sudo rm -rf /srv/tftp
    
    # Reset network adapter to DHCP
    log "Resetting network adapter..."
    sudo ip link set "$adapter" down
    sudo ip addr flush dev "$adapter"
    sudo ip link set "$adapter" up
    sudo dhclient -r "$adapter" &>/dev/null || true
    sudo dhclient "$adapter" &>/dev/null || true
    
    log "Cleanup completed successfully"
}

# Function to setup TFTP server
setup_tftp() {
    local adapter=$1
    local ip_address=$2
    
    log "Setting up TFTP server..."
    
    # Install TFTP server
    log "Installing TFTP server packages..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null 2>&1 || error_exit "Failed to update package lists"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tftpd-hpa >/dev/null 2>&1 || error_exit "Failed to install tftpd-hpa"
    
    # Configure TFTP server
    log "Configuring TFTP server..."
    sudo bash -c "cat > /etc/default/tftpd-hpa" << EOF
TFTP_USERNAME="tftp"
TFTP_DIRECTORY="/srv/tftp"
TFTP_ADDRESS="$ip_address:69"
TFTP_OPTIONS="--secure"
EOF
    
    # Create and set permissions for TFTP directory
    log "Creating TFTP directory and setting permissions..."
    sudo mkdir -p /srv/tftp
    sudo chown -R tftp:tftp /srv/tftp
    sudo chmod -R 777 /srv/tftp
    
    # Restart TFTP service
    log "Restarting TFTP service..."
    sudo systemctl restart tftpd-hpa || error_exit "Failed to restart tftpd-hpa service"
    
    # Check if TFTP service is running
    if ! sudo systemctl is-active --quiet tftpd-hpa; then
        error_exit "TFTP service failed to start"
    fi
    
    log "TFTP server setup completed successfully"
}

# Function to test TFTP server
test_tftp() {
    local ip_address=$1
    
    log "Testing TFTP server..."
    
    # Install TFTP client
    log "Installing TFTP client..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tftp >/dev/null 2>&1 || error_exit "Failed to install TFTP client"
    
    # Create test file
    log "Creating test file..."
    echo "TFTP Test" | sudo tee /srv/tftp/test.txt > /dev/null
    
    # Test TFTP connection
    log "Testing TFTP connection..."
    
    # Create a temporary file for tftp commands
    TFTP_COMMANDS=$(mktemp)
    echo "get test.txt" > "$TFTP_COMMANDS"
    echo "quit" >> "$TFTP_COMMANDS"
    
    # Run tftp with commands from the file
    tftp "$ip_address" < "$TFTP_COMMANDS" || error_exit "TFTP test failed"
    
    # Check if the file was downloaded
    if [ -f "test.txt" ]; then
        log "TFTP test successful - file downloaded"
        rm test.txt
    else
        error_exit "TFTP test failed - file not downloaded"
    fi
    
    # Clean up
    rm "$TFTP_COMMANDS"
    sudo rm /srv/tftp/test.txt
    
    log "TFTP server testing completed successfully"
}

# Function to setup NFS server
setup_nfs() {
    local adapter=$1
    local ip_address=$2
    
    log "Setting up NFS server..."
    
    # Install NFS server
    log "Installing NFS server packages..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null 2>&1 || error_exit "Failed to update package lists"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nfs-kernel-server >/dev/null 2>&1 || error_exit "Failed to install nfs-kernel-server"
    
    # Create shared directory
    log "Creating shared directory..."
    sudo mkdir -p /srv/nfs/shared
    sudo mkdir -p /srv/nfs/shared/petalinux-nfs
    sudo chown nobody:nogroup /srv/nfs/shared
    sudo chmod 777 /srv/nfs/shared
    
    # Configure NFS exports
    log "Configuring NFS exports..."
    sudo bash -c "echo '/srv/nfs/shared $ip_address/24(rw,sync,no_subtree_check,no_root_squash,insecure)' > /etc/exports"
    
    # Export the shared directory
    log "Exporting the shared directory..."
    sudo exportfs -a || error_exit "Failed to export NFS shared directory"
    
    # Restart NFS service
    log "Restarting NFS service..."
    sudo systemctl restart nfs-kernel-server || error_exit "Failed to restart nfs-kernel-server service"
    
    # Check if NFS service is running
    if ! sudo systemctl is-active --quiet nfs-kernel-server; then
        error_exit "NFS service failed to start"
    fi
    
    log "NFS server setup completed successfully"
}

# Function to test NFS server
test_nfs() {
    local ip_address=$1
    
    log "Testing NFS server..."
    
    # Install NFS client
    log "Installing NFS client..."
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nfs-common >/dev/null 2>&1 || error_exit "Failed to install NFS client"
    
    # Create test content
    log "Creating test content..."
    echo "This is a test file." | sudo tee /srv/nfs/shared/testfile.txt > /dev/null
    
    # Create mount point
    log "Creating mount point..."
    sudo mkdir -p /mnt/nfs/test
    
    # Mount NFS share
    log "Mounting NFS share..."
    sudo mount -t nfs "$ip_address:/srv/nfs/shared" /mnt/nfs/test || error_exit "Failed to mount NFS share"
    
    # Verify mount
    if ! df -h | grep -q "/mnt/nfs/test"; then
        sudo umount /mnt/nfs/test 2>/dev/null || true
        error_exit "NFS share not properly mounted"
    fi
    
    # Verify test file is accessible
    if [ ! -f "/mnt/nfs/test/testfile.txt" ]; then
        sudo umount /mnt/nfs/test 2>/dev/null || true
        error_exit "Test file not accessible through NFS mount"
    fi
    
    # Test write access
    log "Testing write access..."
    echo "This is a test write file." | sudo tee /mnt/nfs/test/client_test.txt > /dev/null || {
        sudo umount /mnt/nfs/test 2>/dev/null || true
        error_exit "Failed to write test file to NFS mount"
    }
    
    # Verify the write was successful
    if [ ! -f "/srv/nfs/shared/client_test.txt" ]; then
        sudo umount /mnt/nfs/test 2>/dev/null || true
        error_exit "Written file not found in shared directory"
    fi
    
    # Unmount NFS share
    log "Unmounting NFS share..."
    sudo umount /mnt/nfs/test || error_exit "Failed to unmount NFS share"
    
    # Clean up
    sudo rm -f /srv/nfs/shared/testfile.txt /srv/nfs/shared/client_test.txt
    sudo rmdir /mnt/nfs/test
    
    log "NFS server testing completed successfully"
}

# Function to check service status
check_status() {
    local adapter=$1
    local ip_address
    
    log "Checking services status for adapter $adapter..."
    
    # Get current IP address
    ip_address=$(ip -4 addr show "$adapter" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "No IP")
    
    log "==============================================="
    log "Status Report for $adapter"
    log "Current IP: $ip_address"
    log ""
    
    # Check TFTP status
    log "TFTP Server Status:"
    if systemctl is-active --quiet tftpd-hpa; then
        log "  - TFTP Server: RUNNING"
        log "  - Port: 69 (UDP)"
        log "  - Directory: /srv/tftp"
        if [ -d "/srv/tftp" ]; then
            log "  - TFTP Directory: EXISTS"
        else
            log "  - TFTP Directory: MISSING"
        fi
    else
        log "  - TFTP Server: NOT RUNNING"
    fi
    log ""
    
    # Check NFS status
    log "NFS Server Status:"
    if systemctl is-active --quiet nfs-kernel-server; then
        log "  - NFS Server: RUNNING"
        log "  - Ports: 2049 (TCP/UDP), 111 (portmapper)"
        log "  - Directory: /srv/nfs/shared"
        if [ -d "/srv/nfs/shared" ]; then
            log "  - NFS Directory: EXISTS"
            log "  - Current Exports:"
            exportfs -v | grep "/srv/nfs/shared" || log "    No exports found"
        else
            log "  - NFS Directory: MISSING"
        fi
    else
        log "  - NFS Server: NOT RUNNING"
    fi
    log "==============================================="
}

# Function to display help information
show_help() {
    cat << EOF
Usage: $0 <network_adapter> <command|ip_address>

A script to configure TFTP and NFS servers for direct device connections.

Commands:
    <ip_address>     Configure static IP and setup TFTP/NFS servers
    clean            Remove configuration and restore original state
    status           Display current status of services
    -h, --help       Display this help message

Arguments:
    network_adapter  The network interface to configure (e.g., eth0, enp0s3)
    ip_address      Static IP address to assign (e.g., 192.168.1.100)

Examples:
    $0 eth0 192.168.1.100    Configure eth0 with static IP and setup servers
    $0 eth0 clean            Clean up configuration on eth0
    $0 eth0 status          Show status of services on eth0
    $0 -h                   Show this help message

Service Details:
    TFTP Server:
    - Default Port: 69 (UDP)
    - Root Directory: /srv/tftp
    - Permissions: 777 (rwxrwxrwx)

    NFS Server:
    - Default Ports: 2049 (TCP/UDP), 111 (portmapper)
    - Root Directory: /srv/nfs/shared
    - PetaLinux Directory: /srv/nfs/shared/petalinux-nfs
    - Default Export Options: rw,sync,no_subtree_check

Notes:
    - Script must be run with sudo privileges
    - Clean command will restore DHCP on the adapter
    - Status command shows current IP and service states
    - All operations are logged with timestamps
EOF
}

# Update main function to handle help flag
main() {
    # Check for help flag
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_help
        exit 0
    fi

    # Check if script is run as root
    if [ "$EUID" -ne 0 ]; then
        error_exit "This script must be run as root. Please use sudo."
    fi
    
    # Check if adapter is provided
    if [ -z "$1" ]; then
        error_exit "Please provide a network adapter name and command. Use -h for help."
    fi
    
    local adapter=$1
    
    # Validate adapter
    validate_adapter "$adapter"
    
    # Check for second parameter
    if [ -z "$2" ]; then
        error_exit "Missing second parameter. Use -h for help."
    fi
    
    # Handle different commands
    case "$2" in
        "clean")
            cleanup_services "$adapter"
            log "System restored to original state"
            exit 0
            ;;
        "status")
            check_status "$adapter"
            exit 0
            ;;
        *)
            # Treat as IP address for setup
            local static_ip=$2
            validate_ip "$static_ip"
            
            # Continue with normal setup...
            configure_static_ip "$adapter" "$static_ip"
            setup_tftp "$adapter" "$static_ip"
            test_tftp "$static_ip"
            setup_nfs "$adapter" "$static_ip"
            test_nfs "$static_ip"

            # Set proper ownership of directories
            log "Setting proper ownership of directories..."
            # Get the actual username of the user running sudo
            ACTUAL_USER=$(logname || echo "$SUDO_USER" || whoami)
            log "Setting ownership to user: $ACTUAL_USER"
            sudo chown -R "$ACTUAL_USER":"$ACTUAL_USER" /srv/tftp
            sudo find /srv/nfs/shared/ -maxdepth 1 -exec chown "$ACTUAL_USER":"$ACTUAL_USER" {} \;
            
            log "==============================================="
            log "TFTP and NFS servers setup and testing completed successfully!"
            log "Static IP configured: $static_ip on adapter $adapter"
            log ""
            log "TFTP Server Details:"
            log "  - Server IP: $static_ip:69 (UDP)"
            log "  - Root Directory: /srv/tftp"
            log ""
            log "NFS Server Details:"
            log "  - Server IP: $static_ip"
            log "  - Mount Point: $static_ip:/srv/nfs/shared"
            log "  - Root Directory: /srv/nfs/shared"
            log "  - PetaLinux Directory: /srv/nfs/shared/petalinux-nfs"
            log "  - Default Ports: 2049 (TCP/UDP), 111 (portmapper)"
            log ""
            log "Example NFS Mount Command:"
            log "  sudo mount -t nfs $static_ip:/srv/nfs/shared /mnt/nfs"
            log "==============================================="
            ;;
    esac
}

# Run the main function with command line arguments
main "$@"