#!/bin/bash
# Setup and Run Script for Biomimetic Project
# Auto-activates virtual environment and handles dependencies

echo "Biomimetic Project Setup & Run"
echo "================================="

# Activate virtual environment
echo "Activating virtual environment..."
source /Users/Mubiyn/pulse-env/bin/activate

if [ $? -eq 0 ]; then
    echo "Virtual environment activated: pulse-env"
else
    echo "Failed to activate virtual environment"
    echo "Make sure /Users/Mubiyn/pulse-env/ exists"
    exit 1
fi

# Check if we want to install OpenSourceLeg
if [ "$1" = "install-opensourceleg" ]; then
    echo ""
    echo "Installing OpenSourceLeg SDK..."
    echo "======================================="
    
    # Clone OpenSourceLeg to external directory if not exists
    if [ ! -d "../opensourceleg_external" ]; then
        echo "Cloning OpenSourceLeg repository to external directory..."
        cd ..
        git clone https://github.com/neurobionics/opensourceleg.git opensourceleg_external
        cd unified_biomimetic_project
    else
        echo "OpenSourceLeg repository already exists in external directory"
    fi
    
    # Install OpenSourceLeg from external directory
    echo "Installing OpenSourceLeg SDK from external directory..."
    cd ../opensourceleg_external
    pip install -e .
    cd ../unified_biomimetic_project
    
    echo "OpenSourceLeg installation complete!"
    echo ""
fi

# Show available commands
echo "Available Commands:"
echo "====================="
echo "1. Current biomimetic analysis:"
echo "   python unified_analysis.py"
echo ""
echo "2. Visual-motor integration demo (simulation):"
echo "   python integration_demo.py"
echo ""
echo "3. REAL OpenSourceLeg integration:"
echo "   cd src/integration && python real_opensourceleg_integration.py"
echo ""
echo "4. REAL-TIME Visual-Motor Integration (NEW!):"
echo "   cd src/integration && python realtime_visual_motor_integration.py"
echo ""
echo "5. ADAPTIVE GRASPING - Original Goal! Hand-eye coordination:"
echo "   cd src/integration && python working_adaptive_grasping.py"
echo ""
echo "6. Install OpenSourceLeg SDK:"
echo "   ./setup_and_run.sh install-opensourceleg"
echo ""
echo "7. Individual components:"
echo "   cd src/temporal && python temporal_percept_modeling.py"
echo "   cd src/biological && python biological_variation_modeling.py"
echo "   cd src/evolution && python electrode_evolution_simple.py"
echo ""

# If no arguments, ask what to run
if [ $# -eq 0 ]; then
    echo "❓ What would you like to run? (1-7, or 'q' to quit)"
    read -r choice
    
    case $choice in
        1)
            echo "Running unified biomimetic analysis..."
            python unified_analysis.py
            ;;
        2)
            echo "Running visual-motor integration demo (simulation)..."
            python integration_demo.py
            ;;
        3)
            echo "Running REAL OpenSourceLeg integration..."
            cd src/integration && python real_opensourceleg_integration.py
            ;;
        4)
            echo "Running REAL-TIME Visual-Motor Integration..."
            cd src/integration && python realtime_visual_motor_integration.py
            ;;
        5)
            echo "Running ADAPTIVE GRASPING - Original Goal!"
            cd src/integration && python working_adaptive_grasping.py
            ;;
        6)
            echo "Installing OpenSourceLeg SDK..."
            $0 install-opensourceleg
            ;;
        7)
            echo "Individual components available in src/ directories"
            ;;
        q|Q)
            echo "Goodbye!"
            ;;
        *)
            echo "Invalid choice"
            ;;
    esac
fi 