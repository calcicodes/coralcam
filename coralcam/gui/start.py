#!/usr/bin/env python3
"""
CoralCam Qt5 Application Entry Point
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from .main_window import MainWindow

def main():
    """Main application entry point"""
    # Set Qt platform for Wayland (wlroots)
    if not os.environ.get('QT_QPA_PLATFORM'):
        os.environ['QT_QPA_PLATFORM'] = 'wayland'
    
    # Fallback to xcb if wayland doesn't work
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms'
    
    # Create QApplication with platform fallback
    app = None
    platforms_to_try = ['wayland', 'xcb', 'offscreen']
    
    for platform in platforms_to_try:
        try:
            os.environ['QT_QPA_PLATFORM'] = platform
            app = QApplication(sys.argv)
            print(f"Successfully using Qt platform: {platform}")
            break
        except Exception as e:
            print(f"Failed to use platform {platform}: {e}")
            if app:
                app.quit()
                app = None
            continue
    
    if not app:
        print("Could not initialize Qt with any platform")
        return 1
    
    app.setApplicationName("CoralCam")
    app.setApplicationVersion("1.0.0")
    
    # Set application style for Raspberry Pi
    try:
        app.setStyle('Fusion')
    except:
        print("Warning: Could not set Fusion style")
    
    # Create and show main window
    try:
        window = MainWindow()
        window.show()
        
        # Run the application
        return app.exec_()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())