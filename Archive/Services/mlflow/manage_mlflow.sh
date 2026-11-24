#!/bin/bash

# Function to start the container and measure launch time
start_container() {
    echo "Starting MLflow container..."
    start_time=$(date +%s)
    time docker-compose up -d
    end_time=$(date +%s)
    launch_duration=$((end_time - start_time))
    echo "MLflow container started in $launch_duration seconds. Access it at http://localhost:5000"
}

# Function to stop the container
stop_container() {
    echo "Stopping MLflow container..."
    docker-compose down
    echo "MLflow container stopped."
}

# Function to display usage statistics
show_stats() {
    echo "MLflow Container Usage Statistics:"
    echo "--------------------------------"
    
    # Check if container is running
    if docker-compose ps | grep -q "Up"; then
        echo "Status: Running"
        
        # Get container ID
        container_id=$(docker-compose ps -q mlflow)
        
        # CPU usage
        cpu_usage=$(docker stats --no-stream --format "{{.CPUPerc}}" $container_id)
        echo "CPU Usage: $cpu_usage"
        
        # Memory usage
        mem_usage=$(docker stats --no-stream --format "{{.MemUsage}}" $container_id)
        echo "Memory Usage: $mem_usage"
        
        # Uptime
        uptime=$(docker inspect --format '{{.State.StartedAt}}' $container_id)
        echo "Running since: $uptime"
    else
        echo "Status: Stopped"
    fi
}

# Function to display usage information
usage() {
    echo "Usage: $0 {start|stop|restart|stats}"
    echo "  start   - Start the MLflow container"
    echo "  stop    - Stop the MLflow container"
    echo "  restart - Restart the MLflow container"
    echo "  stats   - Show usage statistics"
}

# Main script logic
case "$1" in
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    restart)
        stop_container
        start_container
        ;;
    stats)
        show_stats
        ;;
    *)
        usage
        exit 1
        ;;
esac

exit 0