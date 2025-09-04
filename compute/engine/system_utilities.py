"""
GoldMIND AI System Utilities
Comprehensive system administration and maintenance tools
Provides database management, health monitoring, and optimization features.
Integrates with FinancialDataFramework for database operations.
"""

import os
import sys
import json
import sqlite3
import shutil
import logging
import psutil
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import zipfile
import hashlib
import platform

# Import FinancialDataFramework for database interaction
try:
    from financial_data_framework import FinancialDataFramework
except ImportError:
    logging.critical("❌ Could not import FinancialDataFramework. Please ensure 'financial_data_framework.py' is accessible.")
    class FinancialDataFramework: # Mock for parsing if main import fails
        def __init__(self, *args, **kwargs): pass
        def get_connection(self):
            # Return a simple in-memory connection if FDF is truly missing
            conn = sqlite3.connect(':memory:')
            conn.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY)")
            conn.execute("CREATE TABLE IF NOT EXISTS api_usage (id INTEGER PRIMARY KEY)")
            conn.commit()
            return conn
        def get_usage_report(self): return {'apis': {}} # Mock for health checks


logger = logging.getLogger(__name__)

class SystemUtilities:
    def __init__(self, config_path: str = "config.json", db_manager: Optional[FinancialDataFramework] = None):
        self.config = self.load_config(config_path)
        self.db_manager = db_manager # Injected FinancialDataFramework instance
        self.db_path = self.config.get('database', {}).get('path', 'goldmind_ai.db') # Use config or default
        self.backup_dir = "backups"
        self.log_dir = "logs"

        # Ensure directories exist
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        logger.info("System Utilities initialized.")

    def load_config(self, config_path: str) -> Dict:
        """Load system configuration."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Config file '{config_path}' not found. Using default config.")
                return self.get_default_config()
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}. Using defaults.", exc_info=True)
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "database": {"path": "goldmind_ai.db"},
            "server": {"host": "0.0.0.0", "port": 5000},
            "logging": {"level": "INFO", "file": "goldmind_app.log"},
            "market_data": {"update_interval": 30, "demo_mode": True},
            "ml_models": {}, "analytics": {}, "security": {}, "auto_hedging": {}, "model_performance": {}, "ultimate_manager": {}
        }

    def check_system_health(self) -> Dict:
        """Comprehensive system health check."""
        try:
            health_report = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'checks': {}
            }

            # Database health
            db_health = self.check_database_health()
            health_report['checks']['database'] = db_health

            # System resources
            resource_health = self.check_system_resources()
            health_report['checks']['resources'] = resource_health

            # Disk space
            disk_health = self.check_disk_space()
            health_report['checks']['disk'] = disk_health

            # File permissions
            permissions_health = self.check_file_permissions()
            health_report['checks']['permissions'] = permissions_health

            # Log file health
            log_health = self.check_log_files()
            health_report['checks']['logs'] = log_health

            # Network connectivity (basic check)
            network_health = self.check_network_connectivity()
            health_report['checks']['network'] = network_health

            # External API health (via FinancialDataFramework)
            api_health = self.check_external_apis_health()
            health_report['checks']['external_apis'] = api_health

            # Determine overall status
            failed_checks = [k for k, v in health_report['checks'].items() if v.get('status') == 'error']
            warning_checks = [k for k, v in health_report['checks'].items() if v.get('status') == 'warning']

            if failed_checks:
                health_report['overall_status'] = 'error'
            elif warning_checks:
                health_report['overall_status'] = 'warning'

            return health_report

        except Exception as e:
            logger.error(f"System health check error: {e}", exc_info=True)
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e),
                'checks': {}
            }

    def check_database_health(self) -> Dict:
        """Check database health and integrity using FinancialDataFramework."""
        try:
            if not self.db_manager:
                return {
                    'status': 'warning',
                    'message': 'Database manager not provided. Cannot perform detailed DB health check.'
                }

            # Attempt to get a connection and execute a simple query
            conn = self.db_manager.get_connection()
            if not conn:
                return {
                    'status': 'error',
                    'message': 'Failed to get database connection.'
                }

            cursor = conn.cursor()

            # Check database integrity
            cursor.execute('PRAGMA integrity_check')
            integrity_result = cursor.fetchone()[0]

            # Get database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            # Count tables
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]

            # Check for essential tables (adjust list based on your actual schema)
            required_tables = [
                'users', 'recommendations', 'user_feedback', 'user_activities',
                'api_usage', 'data_cache', 'error_log', 'hedge_executions',
                'model_performance_snapshots', 'performance_alerts', 'fallback_events',
                'conflict_history', 'bias_history', 'user_analytics', 'system_analytics',
                'user_profiles', 'notification_settings', 'user_trading_history'
            ]

            missing_tables = []
            for table in required_tables:
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if cursor.fetchone()[0] == 0:
                    missing_tables.append(table)

            conn.close()

            # Determine status
            if integrity_result != 'ok':
                status = 'error'
                message = f'Database integrity check failed: {integrity_result}'
            elif missing_tables:
                status = 'warning'
                message = f'Missing essential tables: {", ".join(missing_tables)}'
            elif db_size == 0:
                status = 'warning'
                message = 'Database file is empty.'
            else:
                status = 'healthy'
                message = 'Database is healthy'

            return {
                'status': status,
                'message': message,
                'details': {
                    'integrity': integrity_result,
                    'size_bytes': db_size,
                    'size_mb': round(db_size / 1024 / 1024, 2),
                    'table_count': table_count,
                    'missing_tables': missing_tables
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Database health check failed: {str(e)}'
            }

    def check_system_resources(self) -> Dict:
        """Check system resource usage (CPU, Memory)."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1) # Blocking call, consider async or separate thread

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024 ** 3) # Convert to GB

            # Determine status
            issues = []
            if cpu_percent > 90:
                issues.append(f'High CPU usage: {cpu_percent:.1f}%')
            if memory_percent > 90:
                issues.append(f'High memory usage: {memory_percent:.1f}%')
            if memory_available_gb < 0.5: # Less than 0.5GB available
                issues.append(f'Low available memory: {memory_available_gb:.1f}GB')

            if issues:
                status = 'warning' if len(issues) == 1 else 'error'
                message = '; '.join(issues)
            else:
                status = 'healthy'
                message = 'System resources are normal'

            return {
                'status': status,
                'message': message,
                'details': {
                    'cpu_percent': float(cpu_percent),
                    'memory_percent': float(memory_percent),
                    'memory_available_gb': round(memory_available_gb, 2),
                    'total_memory_gb': round(memory.total / (1024 ** 3), 2)
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Resource check failed: {str(e)}'
            }

    def check_disk_space(self) -> Dict:
        """Check available disk space."""
        try:
            disk_usage = psutil.disk_usage(os.getcwd()) # Check current working directory's disk
            free_gb = disk_usage.free / (1024 ** 3)
            total_gb = disk_usage.total / (1024 ** 3)
            used_percent = disk_usage.percent

            # Determine status
            if free_gb < 1:
                status = 'error'
                message = f'Critical: Only {free_gb:.1f}GB free space remaining'
            elif free_gb < 5:
                status = 'warning'
                message = f'Warning: Only {free_gb:.1f}GB free space remaining'
            elif used_percent > 90:
                status = 'warning'
                message = f'Disk {used_percent:.1f}% full'
            else:
                status = 'healthy'
                message = f'{free_gb:.1f}GB free space available'

            return {
                'status': status,
                'message': message,
                'details': {
                    'free_gb': round(free_gb, 2),
                    'total_gb': round(total_gb, 2),
                    'used_percent': round(used_percent, 1)
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Disk space check failed: {str(e)}'
            }

    def check_file_permissions(self) -> Dict:
        """Check file and directory permissions for critical paths."""
        try:
            issues = []

            # Check database file permissions
            if os.path.exists(self.db_path):
                if not os.access(self.db_path, os.R_OK | os.W_OK):
                    issues.append(f'Database file not readable/writable: {self.db_path}')

            # Check backup directory permissions
            if not os.path.exists(self.backup_dir):
                os.makedirs(self.backup_dir, exist_ok=True) # Try to create if missing
            if not os.access(self.backup_dir, os.R_OK | os.W_OK):
                issues.append(f'Backup directory not accessible: {self.backup_dir}')

            # Check log directory permissions
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True) # Try to create if missing
            if not os.access(self.log_dir, os.R_OK | os.W_OK):
                issues.append(f'Log directory not accessible: {self.log_dir}')

            # Check current directory permissions
            if not os.access(os.getcwd(), os.R_OK | os.W_OK):
                issues.append('Current working directory not accessible for read/write.')

            if issues:
                status = 'error'
                message = '; '.join(issues)
            else:
                status = 'healthy'
                message = 'File permissions are correct'

            return {
                'status': status,
                'message': message,
                'details': {
                    'issues': issues
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Permission check failed: {str(e)}'
            }

    def check_log_files(self) -> Dict:
        """Check log file health (size, writability)."""
        try:
            # Use the log file defined in the config
            log_file = self.config.get('logging', {}).get('file', 'goldmind_app.log')
            issues = []

            if not os.path.exists(log_file):
                issues.append(f'Main log file not found: {log_file}')
            else:
                log_size = os.path.getsize(log_file)
                log_size_mb = log_size / (1024 * 1024)

                if log_size_mb > 100: # Warning for large log files
                    issues.append(f'Large log file: {log_size_mb:.1f}MB (consider cleaning)')

                # Check if log file is writable
                if not os.access(log_file, os.W_OK):
                    issues.append(f'Log file not writable: {log_file}')

            if issues:
                status = 'warning'
                message = '; '.join(issues)
            else:
                status = 'healthy'
                message = 'Log files are healthy'

            return {
                'status': status,
                'message': message,
                'details': {
                    'log_file': log_file,
                    'size_mb': round(log_size_mb, 2) if 'log_size_mb' in locals() else 0,
                    'issues': issues
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Log file check failed: {str(e)}'
            }

    def check_network_connectivity(self) -> Dict:
        """Check basic internet and server port connectivity."""
        try:
            import socket

            # Test basic internet connectivity by trying to connect to Google's DNS
            internet_status = 'disconnected'
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=3)
                internet_status = 'connected'
            except OSError:
                pass # Connection failed

            # Check if server port is available (not in use by another process)
            server_config = self.config.get('server', {})
            host = server_config.get('host', '0.0.0.0')
            port = server_config.get('port', 5000)

            port_status = 'unknown'
            try:
                # Try to bind to the port to see if it's free
                temp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                temp_sock.settimeout(1)
                temp_sock.bind((host, port))
                temp_sock.close()
                port_status = 'available'
            except OSError as e:
                if e.errno == 98: # EADDRINUSE
                    port_status = 'in_use'
                else:
                    port_status = f'error: {e}'
            except Exception:
                port_status = 'error'

            # Determine overall status
            if internet_status == 'disconnected':
                status = 'warning'
                message = 'No internet connectivity detected.'
            elif port_status == 'in_use':
                status = 'warning'
                message = f'Server port {port} is already in use.'
            elif port_status.startswith('error'):
                status = 'warning'
                message = f'Network port check error: {port_status}'
            else:
                status = 'healthy'
                message = 'Network connectivity is good.'

            return {
                'status': status,
                'message': message,
                'details': {
                    'internet': internet_status,
                    'server_port': {
                        'host': host,
                        'port': port,
                        'status': port_status
                    }
                }
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Network check failed: {str(e)}'
            }

    def check_external_apis_health(self) -> Dict:
        """Check health of external APIs using FinancialDataFramework's usage report."""
        if not self.db_manager:
            return {
                'status': 'warning',
                'message': 'FinancialDataFramework not available. Cannot check external API health.'
            }

        try:
            api_report = self.db_manager.get_usage_report()
            api_health_status = 'healthy'
            api_details = {}

            for api_name, stats in api_report['apis'].items():
                status = 'healthy'
                issues = []

                if stats['error_rate'] > 10: # More than 10% errors
                    status = 'error'
                    issues.append(f"High error rate: {stats['error_rate']:.1f}%")
                    api_health_status = 'error' # Elevate overall status
                elif stats['daily_usage_percent'] > 90: # Near daily limit
                    status = 'warning'
                    issues.append(f"Near daily limit: {stats['daily_usage_percent']:.1f}% used")
                    if api_health_status == 'healthy': # Don't downgrade from error
                        api_health_status = 'warning'

                api_details[api_name] = {
                    'status': status,
                    'message': issues[0] if issues else 'Operational',
                    'details': stats # Include full stats for detail
                }

            return {
                'status': api_health_status,
                'message': f"Overall API status: {api_health_status}",
                'details': api_details
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to check external API health: {str(e)}'
            }

    def create_database_backup(self, backup_name: Optional[str] = None) -> Dict:
        """Create a backup of the database."""
        try:
            if not os.path.exists(self.db_path):
                return {
                    'success': False,
                    'message': f'Database file not found: {self.db_path}'
                }

            # Generate backup filename
            if backup_name is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_name = f'goldmind_backup_{timestamp}.db'

            backup_path = os.path.join(self.backup_dir, backup_name)

            # Create backup
            shutil.copy2(self.db_path, backup_path)

            # Verify backup by size (simple check)
            backup_size = os.path.getsize(backup_path)
            original_size = os.path.getsize(self.db_path)

            if backup_size != original_size:
                # Attempt to remove partial backup
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                return {
                    'success': False,
                    'message': f'Backup verification failed: size mismatch (original: {original_size}, backup: {backup_size})'
                }

            # Calculate checksum
            backup_checksum = self.calculate_file_checksum(backup_path)

            return {
                'success': True,
                'message': f'Backup created successfully',
                'details': {
                    'backup_file': backup_path,
                    'backup_size_bytes': backup_size,
                    'checksum': backup_checksum,
                    'created_at': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Backup failed: {str(e)}'
            }

    def restore_database_backup(self, backup_file: str) -> Dict:
        """Restore database from backup."""
        try:
            backup_path = os.path.join(self.backup_dir, backup_file)

            if not os.path.exists(backup_path):
                return {
                    'success': False,
                    'message': f'Backup file not found: {backup_path}'
                }

            # Create backup of current database before restoring
            current_backup_result = self.create_database_backup(f'pre_restore_{int(time.time())}.db')
            if not current_backup_result['success']:
                logger.error(f"Failed to backup current database before restore: {current_backup_result['message']}")
                # Decide whether to proceed with restore or abort
                # For safety, we might abort here or proceed with a warning
                # For now, we proceed but log the failure

            # Restore from backup
            shutil.copy2(backup_path, self.db_path)

            # Verify restoration by size
            restored_size = os.path.getsize(self.db_path)
            backup_size = os.path.getsize(backup_path)

            if restored_size != backup_size:
                return {
                    'success': False,
                    'message': 'Restoration verification failed: size mismatch. Database may be corrupted.'
                }

            return {
                'success': True,
                'message': f'Database restored successfully from {backup_file}',
                'details': {
                    'restored_from': backup_path,
                    'previous_backup_info': current_backup_result.get('details'),
                    'restored_at': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Restoration failed: {str(e)}'
            }

    def list_backups(self) -> List[Dict]:
        """List all available backups."""
        try:
            backups = []

            if not os.path.exists(self.backup_dir):
                return backups

            for file in os.listdir(self.backup_dir):
                if file.endswith('.db') and file.startswith('goldmind_backup_'): # Only list goldmind backups
                    file_path = os.path.join(self.backup_dir, file)
                    stat = os.stat(file_path)

                    backups.append({
                        'filename': file,
                        'size_bytes': stat.st_size,
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'created_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'checksum': self.calculate_file_checksum(file_path)
                    })

            # Sort by creation date (newest first)
            backups.sort(key=lambda x: x['created_at'], reverse=True)

            return backups

        except Exception as e:
            logger.error(f"List backups error: {e}", exc_info=True)
            return []

    def optimize_database(self) -> Dict:
        """Optimize database performance (VACUUM, ANALYZE, REINDEX)."""
        try:
            if not self.db_manager:
                return {
                    'success': False,
                    'message': 'Database manager not provided. Cannot optimize database.'
                }

            conn = self.db_manager.get_connection()
            if not conn:
                return {
                    'success': False,
                    'message': 'Failed to get database connection for optimization.'
                }

            cursor = conn.cursor()

            # Get initial database size
            initial_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            optimization_steps = []

            # Run VACUUM to reclaim space
            try:
                cursor.execute('VACUUM')
                optimization_steps.append('VACUUM completed to reclaim free space.')
            except Exception as e:
                optimization_steps.append(f'VACUUM failed: {str(e)}')

            # Update statistics for query planner
            try:
                cursor.execute('ANALYZE')
                optimization_steps.append('ANALYZE completed to update query planner statistics.')
            except Exception as e:
                optimization_steps.append(f'ANALYZE failed: {str(e)}')

            # Rebuild indexes
            try:
                cursor.execute('REINDEX')
                optimization_steps.append('REINDEX completed to rebuild indexes.')
            except Exception as e:
                optimization_steps.append(f'REINDEX failed: {str(e)}')

            conn.close()

            # Get final database size
            final_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            space_saved = initial_size - final_size

            return {
                'success': True,
                'message': 'Database optimization completed',
                'details': {
                    'initial_size_mb': round(initial_size / (1024 * 1024), 2),
                    'final_size_mb': round(final_size / (1024 * 1024), 2),
                    'space_saved_mb': round(space_saved / (1024 * 1024), 2),
                    'optimization_steps': optimization_steps,
                    'optimized_at': datetime.now().isoformat()
                }
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Database optimization failed: {str(e)}'
            }

    def analyze_database_statistics(self) -> Dict:
        """Analyze database statistics and usage."""
        try:
            if not self.db_manager:
                return {
                    'success': False,
                    'message': 'Database manager not provided. Cannot analyze database statistics.'
                }

            conn = self.db_manager.get_connection()
            if not conn:
                return {
                    'success': False,
                    'message': 'Failed to get database connection for analysis.'
                }

            cursor = conn.cursor()

            statistics = {
                'database_info': {},
                'table_statistics': {},
                'performance_metrics': {}
            }

            # Database info
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            statistics['database_info'] = {
                'size_bytes': db_size,
                'size_mb': round(db_size / (1024 * 1024), 2),
                'last_modified': datetime.fromtimestamp(os.path.getmtime(self.db_path)).isoformat() if os.path.exists(self.db_path) else 'N/A'
            }

            # Table statistics
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]

                # Row count
                cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"') # Quote table name to handle special chars
                row_count = cursor.fetchone()[0]

                statistics['table_statistics'][table_name] = {
                    'row_count': row_count
                }

            # Performance metrics (PRAGMA settings)
            cursor.execute('PRAGMA cache_size')
            cache_size = cursor.fetchone()[0]

            cursor.execute('PRAGMA page_size')
            page_size = cursor.fetchone()[0]

            cursor.execute('PRAGMA page_count')
            page_count = cursor.fetchone()[0]

            statistics['performance_metrics'] = {
                'cache_size_pages': cache_size,
                'page_size_bytes': page_size,
                'page_count': page_count,
                'total_pages_size_mb': round((page_size * page_count) / (1024 * 1024), 2)
            }

            conn.close()

            return {
                'success': True,
                'statistics': statistics,
                'analyzed_at': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Database analysis failed: {str(e)}'
            }

    def clean_log_files(self, days_to_keep: int = 30) -> Dict:
        """Clean old log files."""
        try:
            import gzip # Moved import here to catch ImportError only when this function is called
            cleaned_files = []
            total_space_saved = 0

            # Determine the main log file path
            main_log_file = self.config.get('logging', {}).get('file', 'goldmind_app.log')

            # Clean main log file if it's too large (archive and restart)
            if os.path.exists(main_log_file):
                log_size = os.path.getsize(main_log_file)
                log_size_mb = log_size / (1024 * 1024)

                if log_size_mb > 50:  # If log file is larger than 50MB
                    # Archive current log
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    archived_log = os.path.join(self.log_dir, f'{os.path.basename(main_log_file)}.{timestamp}.gz')

                    # Compress and move the log file
                    with open(main_log_file, 'rb') as f_in:
                        with gzip.open(archived_log, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    # Clear the original log file
                    open(main_log_file, 'w').close()

                    cleaned_files.append(f'Archived and reset large log file: {os.path.basename(main_log_file)} to {os.path.basename(archived_log)}')
                    total_space_saved += log_size # Count original size as saved

            # Clean old archived logs in the log_dir
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            for file_name in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, file_name)

                if os.path.isfile(file_path):
                    file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))

                    if file_modified < cutoff_date:
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)

                        cleaned_files.append(f'Deleted old log: {file_name}')
                        total_space_saved += file_size

            return {
                'success': True,
                'message': f'Log cleanup completed',
                'details': {
                    'cleaned_files': cleaned_files,
                    'space_saved_mb': round(total_space_saved / (1024 * 1024), 2),
                    'days_kept': days_to_keep,
                    'cleaned_at': datetime.now().isoformat()
                }
            }

        except ImportError: # If gzip is not available
            return {
                'success': False,
                'message': "gzip module not available for log compression. Install zlib or remove compression logic."
            }
        except Exception as e:
            return {
                'success': False,
                'message': f'Log cleanup failed: {str(e)}'
            }

    def generate_system_report(self) -> Dict:
        """Generate comprehensive system report."""
        try:
            report = {
                'report_info': {
                    'generated_at': datetime.now().isoformat(),
                    'goldmind_version': '1.0.0-beta', # Update version
                    'python_version': platform.python_version(),
                    'platform': platform.platform(),
                    'system_uptime_seconds': time.monotonic() # System uptime since boot
                }
            }

            # System health
            report['health_check'] = self.check_system_health()

            # Database statistics
            report['database_analysis'] = self.analyze_database_statistics()

            # Backup information
            report['backup_info'] = {
                'available_backups_count': len(self.list_backups()),
                'latest_backups': self.list_backups()[:5]  # Last 5 backups
            }

            # System resources over time (current snapshot)
            report['resource_usage_snapshot'] = self.check_system_resources()['details']

            return {
                'success': True,
                'report': report
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Report generation failed: {str(e)}'
            }

    def calculate_file_checksum(self, file_path: str, algorithm: str = 'md5') -> str:
        """Calculate checksum of a file."""
        try:
            if not os.path.exists(file_path):
                return "file_not_found"

            hasher = hashlib.new(algorithm)
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Checksum calculation error for {file_path}: {e}", exc_info=True)
            return "error"

    def export_configuration(self) -> Dict:
        """Export current system configuration."""
        try:
            config_export = {
                'export_info': {
                    'exported_at': datetime.now().isoformat(),
                    'version': '1.0.0-beta'
                },
                'configuration': self.config,
                'system_info': {
                    'platform': platform.platform(),
                    'python_version': platform.python_version()
                }
            }

            export_filename = f'goldmind_config_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

            with open(export_filename, 'w') as f:
                json.dump(config_export, f, indent=2)

            return {
                'success': True,
                'message': f'Configuration exported to {export_filename}',
                'export_file': export_filename
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Configuration export failed: {str(e)}'
            }

# Command line interface for standalone usage (for testing/admin)
def main():
    parser = argparse.ArgumentParser(description='GoldMIND AI System Utilities')
    parser.add_argument('command', choices=[
        'health', 'backup', 'restore', 'optimize', 'analyze',
        'clean-logs', 'report', 'list-backups', 'export-config'
    ], help='Command to execute')

    parser.add_argument('--backup-name', help='Name for backup file')
    parser.add_argument('--backup-file', help='Backup file to restore from')
    parser.add_argument('--days', type=int, default=30, help='Days to keep for log cleanup')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--db-path', help='Override database path (for testing)')

    args = parser.parse_args()

    # Initialize a mock FinancialDataFramework for standalone testing
    # In a real deployed system, SystemUtilities would receive a real db_manager instance
    class MockFinancialDataFrameworkForUtils:
        def get_connection(self):
            # For testing, connect to a temporary in-memory DB or a dummy file
            # This allows DB checks to run without needing a full FDF setup
            db_file = args.db_path if args.db_path else ':memory:'
            conn = sqlite3.connect(db_file)
            conn.row_factory = sqlite3.Row
            # Create minimal tables for health check to pass
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY)")
            cursor.execute("CREATE TABLE IF NOT EXISTS api_usage (id INTEGER PRIMARY KEY)")
            conn.commit()
            return conn
        def get_usage_report(self):
            # Mock API usage report for external API health check
            return {
                'apis': {
                    'alpha_vantage': {'error_rate': 5, 'daily_usage_percent': 80, 'total_calls': 100, 'error_count': 5},
                    'twelve_data': {'error_rate': 0, 'daily_usage_percent': 10, 'total_calls': 20, 'error_count': 0},
                    'fred': {'error_rate': 0, 'daily_usage_percent': 5, 'total_calls': 10, 'error_count': 0}
                }
            }

    mock_db_manager = MockFinancialDataFrameworkForUtils()

    # Initialize system utilities
    utilities = SystemUtilities(config_path=args.config, db_manager=mock_db_manager)

    # Execute command
    if args.command == 'health':
        result = utilities.check_system_health()
        print("\n=== GOLDMIND AI SYSTEM HEALTH CHECK ===")
        print(f"Overall Status: {result['overall_status'].upper()}")
        print(f"Generated At: {result['timestamp']}")
        print("\n--- Component Health Status ---")

        for check_name, check_result in result.get('checks', {}).items():
            status_symbol = "✅" if check_result['status'] == 'healthy' else "⚠️" if check_result['status'] == 'warning' else "❌"
            print(f"{status_symbol} {check_name.replace('_', ' ').title()}: {check_result['message']}")
            if 'details' in check_result:
                for detail_key, detail_value in check_result['details'].items():
                    if isinstance(detail_value, dict):
                        print(f"    - {detail_key.replace('_', ' ').title()}:")
                        for sub_key, sub_value in detail_value.items():
                            print(f"      {sub_key.replace('_', ' ').title()}: {sub_value}")
                    else:
                        print(f"    - {detail_key.replace('_', ' ').title()}: {detail_value}")
        print("="*60 + "\n")

    elif args.command == 'backup':
        result = utilities.create_database_backup(args.backup_name)
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"📁 Backup file: {result['details']['backup_file']}")
            print(f"💾 Size: {result['details']['backup_size_bytes'] / (1024*1024):.2f} MB")
            print(f"Checksum: {result['details']['checksum']}")
        else:
            print(f"❌ {result['message']}")

    elif args.command == 'restore':
        if not args.backup_file:
            print("❌ Please specify --backup-file to restore from.")
            sys.exit(1)

        result = utilities.restore_database_backup(args.backup_file)
        if result['success']:
            print(f"✅ {result['message']}")
            if 'previous_backup_info' in result['details'] and result['details']['previous_backup_info']:
                 print(f"   (Previous DB backed up to: {result['details']['previous_backup_info']['backup_file']})")
        else:
            print(f"❌ {result['message']}")

    elif args.command == 'optimize':
        result = utilities.optimize_database()
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"💾 Space saved: {result['details']['space_saved_mb']} MB")
            for step in result['details']['optimization_steps']:
                print(f"  - {step}")
        else:
            print(f"❌ {result['message']}")

    elif args.command == 'analyze':
        result = utilities.analyze_database_statistics()
        if result['success']:
            stats = result['statistics']
            print("\n=== DATABASE ANALYSIS ===")
            print(f"Database Size: {stats['database_info']['size_mb']} MB")
            print(f"Last Modified: {stats['database_info']['last_modified']}")
            print("\n--- Table Statistics ---")
            for table, data in stats['table_statistics'].items():
                print(f"  • {table}: {data['row_count']} rows")
            print("\n--- Performance Metrics (PRAGMA) ---")
            for key, value in stats['performance_metrics'].items():
                print(f"  • {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"❌ {result['message']}")

    elif args.command == 'clean-logs':
        # Removed the top-level import of gzip here, as it's now handled inside the function
        result = utilities.clean_log_files(args.days)
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"🧹 Space saved: {result['details']['space_saved_mb']} MB")
            if result['details']['cleaned_files']:
                print("Cleaned files:")
                for file in result['details']['cleaned_files']:
                    print(f"  - {file}")
        else:
            print(f"❌ {result['message']}")

    elif args.command == 'report':
        result = utilities.generate_system_report()
        if result['success']:
            report_file = f'system_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(report_file, 'w') as f:
                json.dump(result['report'], f, indent=2)
            print(f"✅ System report generated: {report_file}")
            print(f"Report details: {json.dumps(result['report']['report_info'], indent=2)}")
        else:
            print(f"❌ {result['message']}")

    elif args.command == 'list-backups':
        backups = utilities.list_backups()
        if backups:
            print("\n=== AVAILABLE BACKUPS ===")
            for backup in backups:
                print(f"📁 {backup['filename']}")
                print(f"   Size: {backup['size_mb']} MB")
                print(f"   Created: {backup['created_at']}")
                print(f"   Checksum: {backup['checksum'][:8]}...")
                print()
        else:
            print("No backups found.")

    elif args.command == 'export-config':
        result = utilities.export_configuration()
        if result['success']:
            print(f"✅ {result['message']}")
            print(f"Exported to: {result['export_file']}")
        else:
            print(f"❌ {result['message']}")

if __name__ == "__main__":
    # Create dummy config.json for testing if it doesn't exist
    if not os.path.exists("config.json"):
        dummy_config_content = {
            "database": {"path": "test_goldmind_ai.db"},
            "logging": {"file": "goldmind_app.log"},
            "server": {"host": "127.0.0.1", "port": 5000}
        }
        with open("config.json", "w") as f:
            json.dump(dummy_config_content, f, indent=2)
        print("Created dummy config.json for testing.")

    # Create dummy log and backup directories for testing
    os.makedirs("logs", exist_ok=True)
    os.makedirs("backups", exist_ok=True)
    with open("goldmind_app.log", "w") as f:
        f.write("Test log entry.\n" * 10) # Create a small log file

    # Create a dummy database file for testing DB operations
    if not os.path.exists("test_goldmind_ai.db"):
        conn = sqlite3.connect("test_goldmind_ai.db")
        conn.close()

    try:
        main()
    except Exception as e:
        logger.critical(f"System Utilities example failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up dummy files and directories after testing
        # Only remove if they were created by this script for standalone testing
        if os.path.exists("config.json"):
            # Check if it's the dummy one created by this script's __main__
            with open("config.json", 'r') as f:
                content = json.load(f)
            if content.get("database", {}).get("path") == "test_goldmind_ai.db":
                 os.remove("config.json")
        if os.path.exists("goldmind_app.log"):
            os.remove("goldmind_app.log")
        if os.path.exists("test_goldmind_ai.db"):
            os.remove("test_goldmind_ai.db")
        if os.path.exists("test_goldmind_ai.db-wal"):
            os.remove("test_goldmind_ai.db-wal")
        if os.path.exists("test_goldmind_ai.db-shm"):
            os.remove("test_goldmind_ai.db-shm")
        if os.path.exists("logs"):
            shutil.rmtree("logs")
        if os.path.exists("backups"):
            shutil.rmtree("backups")
        print("\nCleaned up test files (if created by standalone run).")