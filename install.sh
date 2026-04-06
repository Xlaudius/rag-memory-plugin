#!/bin/bash
# RAG Memory Plugin Installation Script
# Automates installation with proper error handling and environment detection

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║       RAG Memory Plugin - Installation                    ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${CYAN}→ $1${NC}"
}

check_python_version() {
    print_info "Checking Python version..."

    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        echo "  Please install Python 3.10 or higher"
        exit 1
    fi

    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
        print_error "Python $PYTHON_VERSION is too old"
        echo "  Required: Python 3.10 or higher"
        exit 1
    fi

    print_success "Python $PYTHON_VERSION detected"
    echo ""
}

check_pip() {
    print_info "Checking pip..."

    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed"
        echo "  Install with: sudo apt install python3-pip"
        exit 1
    fi

    print_success "pip3 is available"
    echo ""
}

detect_environment() {
    print_info "Detecting Python environment..."

    # Test for externally-managed-environment
    if pip3 install --dry-run --user test-package-xyz-123 2>&1 | grep -q "externally-managed-environment"; then
        print_warning "Externally managed Python environment detected"
        return 1
    else
        print_success "User-space installation available"
        return 0
    fi
}

install_user_space() {
    print_info "Installing to user space..."

    if pip3 install --user 'git+https://github.com/favouraka/rag-memory-plugin.git#egg=rag-memory-plugin[neural]'; then
        print_success "Installation completed"
        return 0
    else
        return 1
    fi
}

install_virtual_env() {
    print_info "Setting up virtual environment..."

    VENV_PATH="$HOME/.hermes-venv"

    if [ -d "$VENV_PATH" ]; then
        print_warning "Virtual environment already exists at $VENV_PATH"
        if ! confirm "Recreate virtual environment?"; then
            print_info "Using existing virtual environment"
        else
            print_info "Removing old virtual environment..."
            rm -rf "$VENV_PATH"
        fi
    fi

    if [ ! -d "$VENV_PATH" ]; then
        print_info "Creating virtual environment at $VENV_PATH..."
        python3 -m venv "$VENV_PATH"
    fi

    print_info "Installing package to virtual environment..."
    source "$VENV_PATH/bin/activate"

    if pip install 'git+https://github.com/favouraka/rag-memory-plugin.git#egg=rag-memory-plugin[neural]'; then
        print_success "Installation completed"

        # Add to PATH if not already there
        if ! grep -q "$VENV_PATH/bin" "$HOME/.bashrc" 2>/dev/null; then
            print_info "Adding virtual environment to PATH..."
            echo "" >> "$HOME/.bashrc"
            echo "# RAG Memory Plugin" >> "$HOME/.bashrc"
            echo "export PATH=\"$VENV_PATH/bin:\$PATH\"" >> "$HOME/.bashrc"
            print_success "Added to ~/.bashrc"
            print_warning "Run 'source ~/.bashrc' or restart your terminal"
        fi

        deactivate
        return 0
    else
        return 1
    fi
}

install_break_system() {
    print_warning "This will install to system Python"
    print_warning "This may break system package updates"
    echo ""

    if ! confirm "Continue with --break-system-packages?"; then
        return 1
    fi

    print_info "Installing with --break-system-packages..."

    if pip3 install --break-system-packages 'git+https://github.com/favouraka/rag-memory-plugin.git#egg=rag-memory-plugin[neural]'; then
        print_success "Installation completed"
        return 0
    else
        return 1
    fi
}

confirm() {
    if [ -t 0 ]; then
        read -p "$1 (y/N): " response
        response=$(echo "$response" | tr '[:upper:]' '[:lower:]')
        [[ "$response" =~ ^yes|y$ ]]
    else
        return 1
    fi
}

verify_installation() {
    print_info "Verifying installation..."

    # Check if command is available
    if command -v rag-memory &> /dev/null; then
        print_success "rag-memory command is available"
    else
        print_warning "rag-memory command not found in PATH"
        print_info "You may need to run: source ~/.bashrc"
        return 1
    fi

    # Run doctor to verify
    if rag-memory doctor &> /dev/null; then
        print_success "Package is working correctly"
    else
        print_warning "Package may not be fully functional"
        print_info "Run 'rag-memory doctor' for details"
    fi

    echo ""
}

run_setup() {
    print_info "Running setup..."

    if command -v rag-memory &> /dev/null; then
        rag-memory setup
    else
        print_warning "rag-memory command not available yet"
        print_info "Please run 'rag-memory setup' after installation"
    fi
}

show_next_steps() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                   Installation Complete!                   ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${GREEN}Next Steps:${NC}"
    echo ""
    echo "  1. Run setup wizard:"
    echo -e "     ${CYAN}rag-memory setup${NC}"
    echo ""
    echo "  2. Check installation:"
    echo -e "     ${CYAN}rag-memory doctor${NC}"
    echo ""
    echo "  3. Search memory:"
    echo -e "     ${CYAN}rag-memory search \"your query\"${NC}"
    echo ""
    echo "  4. For help:"
    echo -e "     ${CYAN}rag-memory --help${NC}"
    echo ""
    echo -e "${CYAN}Documentation: https://github.com/favouraka/rag-memory-plugin${NC}"
    echo ""
}

# Main installation flow
main() {
    print_header

    # Check prerequisites
    check_python_version
    check_pip

    # Detect environment and try installation methods
    if detect_environment; then
        # User-space installation available
        print_info "Attempting user-space installation..."
        if install_user_space; then
            verify_installation
            run_setup
            show_next_steps
            exit 0
        fi
    fi

    # User-space failed or not available, try virtual env
    print_info "User-space installation not available or failed"
    print_info "Trying virtual environment..."
    if install_virtual_env; then
        verify_installation
        show_next_steps
        exit 0
    fi

    # Virtual env failed, offer break-system-packages
    print_info "Virtual environment installation failed"
    print_warning "Last resort: --break-system-packages"
    echo ""
    if install_break_system; then
        verify_installation
        show_next_steps
        exit 0
    fi

    # All methods failed
    print_error "Installation failed"
    echo ""
    echo "Please report this issue:"
    echo "https://github.com/favouraka/rag-memory-plugin/issues"
    exit 1
}

# Run main function
main "$@"
