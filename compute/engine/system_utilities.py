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
    # --- IMPORTANT CHANGE HERE: Now accepts 'config' (dict) directly ---
    def __init__(self, config: Dict, db_manager: Optional[FinancialDataFramework] = None):
        # The 'config' argument now directly receives the resolved configuration dictionary
        self.config = config
        self.db_manager = db_manager # Injected FinancialDataFramework instance
        
        # Access config values directly from the passed 'config' dictionary
        # No need to load config from a path here, as it's already resolved upstream
        self.db_path = self.config.get('database', {}).get('path', './data/goldmind_ai.db') # Use config or default
        self.backup_dir = "backups"
        self.log_dir = "logs"

        # Ensure directories exist (based on local paths or paths from config)
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else './data', exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        logger.info("System Utilities initialized.")

    # Removed load_config and get_default_config methods as config is now passed directly
    # These methods are no longer needed within SystemUtilities itself for its primary function.
    # The responsibility of loading and resolving config is now handled by ProductionConfigManager and entrypoint.

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
            # Use PRAGMA integrity_check if it's an SQLite connection, otherwise adjust for other DBs
            try:
                cursor.execute('PRAGMA integrity_check')
                integrity_result = cursor.fetchone()[0]
            except sqlite3.OperationalError: # Catch if not SQLite (e.g., mock for other DBs)
                integrity_result = 'N/A (not SQLite or connection error)'

            # Get database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0

            # Count tables
            try:
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                table_count = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                table_count = 'N/A'


            # Check for essential tables (adjust list based on your actual schema)
            required_tables = [
                'users', 'recommendations', 'user_feedback', 'user_activities',
                'api_usage', 'data_cache', 'error_log', 'hedge_executions',
                'model_performance_snapshots', 'performance_alerts', 'fallback_events',
                'conflict_history', 'bias_history', 'user_analytics', 'system_analytics',
                'user_profiles', 'notification_settings', 'user_trading_history'
            ]

            missing_tables = []
            if isinstance(table_count, int): # Only check if we could count tables
                for table in required_tables:
                    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table,))
                    if cursor.fetchone()[0] == 0:
                        missing_tables.append(table)

            conn.close()

            # Determine status
            if integrity_result != 'ok' and integrity_result != 'N/A (not SQLite or connection error)':
                status = 'error'
                message = f'Database integrity check failed: {integrity_result}'
            elif missing_tables:
                status = 'warning'
                message = f'Missing essential tables: {", ".join(missing_tables)}'
            elif db_size == 0 and os.path.exists(self.db_path):
                status = 'warning'
                message = 'Database file is empty.'
            elif not os.path.exists(self.db_path):
                status = 'error'
                message = f'Database file not found at expected path: {self.db_path}'
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
            if cpu_percent > self.config.get('system', {}).get('cpu_usage_threshold', 90): # Use config for thresholds
                issues.append(f'High CPU usage: {cpu_percent:.1f}%')
            if memory_percent > self.config.get('system', {}).get('memory_usage_threshold', 90): # Use config for thresholds
                issues.append(f'High memory usage: {memory_percent:.1f}%')
            if memory_available_gb < self.config.get('system', {}).get('min_available_memory_gb', 0.5): # Use config for thresholds
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
            # Check the directory where the database is stored
            check_path = os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else os.getcwd()
            disk_usage = psutil.disk_usage(check_path)
            free_gb = disk_usage.free / (1024 ** 3)
            total_gb = disk_usage.total / (1024 ** 3)
            used_percent = disk_usage.percent

            # Determine status using thresholds from config
            min_free_gb_critical = self.config.get('system', {}).get('disk_free_gb_critical', 1.0)
            min_free_gb_warning = self.config.get('system', {}).get('disk_free_gb_warning', 5.0)
            max_used_percent_warning = self.config.get('system', {}).get('disk_used_percent_warning', 90.0)


            if free_gb < min_free_gb_critical:
                status = 'error'
                message = f'Critical: Only {free_gb:.1f}GB free space remaining on {check_path}'
            elif free_gb < min_free_gb_warning:
                status = 'warning'
                message = f'Warning: Only {free_gb:.1f}GB free space remaining on {check_path}'
            elif used_percent > max_used_percent_warning:
                status = 'warning'
                message = f'Disk {used_percent:.1f}% full on {check_path}'
            else:
                status = 'healthy'
                message = f'{free_gb:.1f}GB free space available on {check_path}'

            return {
                'status': status,
                'message': message,
                'details': {
                    'checked_path': check_path,
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

            # Paths to check based on configuration
            paths_to_check = [
                self.db_path,
                self.backup_dir,
                self.log_dir,
                os.getcwd(), # Current working directory
                # Add other critical directories from config, e.g., model storage
                self.config.get('ml_models', {}).get('model_storage_path', 'lstm_models')
            ]
            
            # Ensure model storage path exists for check
            model_storage_path = self.config.get('ml_models', {}).get('model_storage_path', 'lstm_models')
            os.makedirs(model_storage_path, exist_ok=True)


            for path_to_check in paths_to_check:
                if not os.path.exists(path_to_check):
                    issues.append(f'Path not found: {path_to_check}')
                    continue

                # Check read/write permissions
                is_readable = os.access(path_to_check, os.R_OK)
                is_writable = os.access(path_to_check, os.W_OK)

                if not is_readable:
                    issues.append(f'Path not readable: {path_to_check}')
                if not is_writable:
                    issues.append(f'Path not writable: {path_to_check}')

            if issues:
                status = 'error'
                message = '; '.join(issues)
            else:
                status = 'healthy'
                message = 'File permissions are correct for critical paths'

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
            # Use the log file path defined in the config, ensuring it's relative to log_dir if not absolute
            log_file_name = self.config.get('logging', {}).get('file', 'goldmind_app.log')
            log_file_path = os.path.join(self.log_dir, log_file_name) # Assume logs are in log_dir

            issues = []

            if not os.path.exists(log_file_path):
                issues.append(f'Main log file not found: {log_file_path}')
            else:
                log_size = os.path.getsize(log_file_path)
                log_size_mb = log_size / (1024 * 1024)

                # Use log file size threshold from config if available, otherwise default
                log_size_threshold_mb = self.config.get('logging', {}).get('max_log_size_mb', 100)
                if log_size_mb > log_size_threshold_mb:
                    issues.append(f'Large log file: {log_file_path} is {log_size_mb:.1f}MB (threshold: {log_size_threshold_mb}MB)')

                # Check if log file is writable
                if not os.access(log_file_path, os.W_OK):
                    issues.append(f'Log file not writable: {log_file_path}')

            # Also check if the log directory itself is writable
            if not os.access(self.log_dir, os.W_OK):
                issues.append(f'Log directory not writable: {self.log_dir}')

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
                    'log_file_path': log_file_path,
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
                # Use a small timeout for a quick check
                socket.create_connection(("8.8.8.8", 53), timeout=1).close()
                internet_status = 'connected'
            except OSError:
                pass # Connection failed
            except Exception as e:
                logger.debug(f"Internet connectivity check error: {e}")
                
            # Check if server port is available (not in use by another process)
            server_config = self.config.get('server', {})
            host = server_config.get('host', '0.0.0.0')
            port = server_config.get('port', 5000)

            port_status = 'unknown'
            if host == '0.0.0.0': # Can't bind to 0.0.0.0 reliably for checking if it's *in use* by *another* process
                # If host is 0.0.0.0, we can't reliably check if it's already bound without trying to bind on a specific IP.
                # This check is more reliable for specific hosts (e.g., '127.0.0.1').
                port_status = 'bindable (0.0.0.0 host)'
            else:
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
                except Exception as e:
                    logger.debug(f"Server port check error: {e}")
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

            # Use thresholds from config for API error rates
            api_error_threshold = self.config.get('analytics', {}).get('api_error_rate_threshold_percent', 10) # Default 10%
            api_usage_warning_threshold = self.config.get('analytics', {}).get('api_usage_warning_percent', 90) # Default 90%

            for api_name, stats in api_report['apis'].items():
                status = 'healthy'
                issues = []

                # Ensure 'error_rate' and 'daily_usage_percent' exist in stats
                current_error_rate = stats.get('error_rate', 0)
                current_daily_usage_percent = stats.get('daily_usage_percent', 0)

                if current_error_rate > api_error_threshold:
                    status = 'error'
                    issues.append(f"High error rate: {current_error_rate:.1f}%")
                    api_health_status = 'error' # Elevate overall status
                elif current_daily_usage_percent > api_usage_warning_threshold:
                    status = 'warning'
                    issues.append(f"Near daily limit: {current_daily_usage_percent:.1f}% used")
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
            # Assume log files defined in config are within the self.log_dir
            main_log_file_name = self.config.get('logging', {}).get('file', 'goldmind_app.log')
            main_log_file_path = os.path.join(self.log_dir, main_log_file_name)


            # Clean main log file if it's too large (archive and restart)
            if os.path.exists(main_log_file_path):
                log_size = os.path.getsize(main_log_file_path)
                # Use log file size threshold from config
                log_size_threshold_mb = self.config.get('logging', {}).get('max_log_size_mb', 50)
                log_size_threshold_bytes = log_size_threshold_mb * 1024 * 1024

                if log_size > log_size_threshold_bytes:
                    # Archive current log
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    archived_log = os.path.join(self.log_dir, f'{os.path.basename(main_log_file_path)}.{timestamp}.gz')

                    # Compress and move the log file
                    with open(main_log_file_path, 'rb') as f_in:
                        with gzip.open(archived_log, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    # Clear the original log file
                    open(main_log_file_path, 'w').close()

                    cleaned_files.append(f'Archived and reset large log file: {os.path.basename(main_log_file_path)} to {os.path.basename(archived_log)}')
                    total_space_saved += log_size # Count original size as saved

            # Clean old archived logs in the log_dir
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            for file_name in os.listdir(self.log_dir):
                file_path = os.path.join(self.log_dir, file_name)

                if os.path.isfile(file_path):
                    file_modified = datetime.fromtimestamp(os.path.getmtime(file_path))

                    if file_modified < cutoff_date and file_name != main_log_file_name: # Don't delete the currently active log file if it's within days_to_keep
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
                    'goldmind_version': self.config.get('system', {}).get('version', '1.0.0-beta'), # Use version from config
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
                    'version': self.config.get('system', {}).get('version', '1.0.0-beta')
                },
                'configuration': self.config, # Export the full resolved config
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
    # Removed --config argument as SystemUtilities no longer loads it directly
    parser.add_argument('--db-path', default='test_goldmind_ai.db', help='Override database path for testing (e.g., test_goldmind_ai.db)')

    args = parser.parse_args()

    # Define a default minimal config for standalone CLI execution
    # This acts as a fallback if a full config from entrypoint isn't available
    default_cli_config = {
        "database": {"path": args.db_path}, # Use the CLI provided db_path
        "server": {"host": "0.0.0.0", "port": 5000},
        "logging": {"level": "INFO", "file": "goldmind_app.log", "max_log_size_mb": 50},
        "system": {
            "environment": "development",
            "cpu_usage_threshold": 90,
            "memory_usage_threshold": 90,
            "min_available_memory_gb": 0.5,
            "disk_free_gb_critical": 1.0,
            "disk_free_gb_warning": 5.0,
            "disk_used_percent_warning": 90.0,
            "version": "1.0.0-beta"
        },
        "ml_models": {"model_storage_path": "lstm_models"}, # Add model storage path for permissions check
        "analytics": {"api_error_rate_threshold_percent": 10, "api_usage_warning_percent": 90},
    }

    # Initialize a mock FinancialDataFramework for standalone testing
    class MockFinancialDataFrameworkForUtils:
        def get_connection(self):
            # For testing, connect to a temporary in-memory DB or a dummy file
            db_file_path = default_cli_config['database']['path']
            # Ensure the directory for the dummy DB exists during standalone test
            os.makedirs(os.path.dirname(db_file_path) if os.path.dirname(db_file_path) else '.', exist_ok=True)
            
            conn = sqlite3.connect(db_file_path)
            conn.row_factory = sqlite3.Row
            # Create minimal tables for health check to pass
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY)")
            cursor.execute("CREATE TABLE IF NOT EXISTS api_usage (id INTEGER PRIMARY KEY)")
            # Add essential tables for check_database_health to pass as healthy
            required_tables_for_mock = [
                'recommendations', 'user_feedback', 'user_activities',
                'data_cache', 'error_log', 'hedge_executions',
                'model_performance_snapshots', 'performance_alerts', 'fallback_events',
                'conflict_history', 'bias_history', 'user_analytics', 'system_analytics',
                'user_profiles', 'notification_settings', 'user_trading_history'
            ]
            for table in required_tables_for_mock:
                cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY)")
            
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

    # Initialize system utilities with the default CLI config
    utilities = SystemUtilities(config=default_cli_config, db_manager=mock_db_manager)

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
    # Configure logging for standalone usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define paths for standalone testing
    test_db_dir = "data"
    test_db_path = os.path.join(test_db_dir, "test_goldmind_ai.db")
    test_log_dir = "logs"
    test_log_file = os.path.join(test_log_dir, "goldmind_app.log")
    test_backup_dir = "backups"
    test_lstm_model_dir = "lstm_models" # Added for permission checks

    # Clean up dummy files and directories from previous runs to ensure clean test environment
    # Be careful with shutil.rmtree in production scenarios; this is for development/testing
    print("Cleaning up previous test artifacts...")
    if os.path.exists("config.json"): os.remove("config.json")
    if os.path.exists(test_log_file): os.remove(test_log_file)
    if os.path.exists(test_db_path): os.remove(test_db_path)
    if os.path.exists(f"{test_db_path}-wal"): os.remove(f"{test_db_path}-wal")
    if os.path.exists(f"{test_db_path}-shm"): os.remove(f"{test_db_path}-shm")
    if os.path.exists(test_log_dir): shutil.rmtree(test_log_dir)
    if os.path.exists(test_backup_dir): shutil.rmtree(test_backup_dir)
    if os.path.exists(test_lstm_model_dir): shutil.rmtree(test_lstm_model_dir) # Clean model dir

    # Re-create necessary dummy directories for the test run
    os.makedirs(test_db_dir, exist_ok=True)
    os.makedirs(test_log_dir, exist_ok=True)
    os.makedirs(test_backup_dir, exist_ok=True)
    os.makedirs(test_lstm_model_dir, exist_ok=True) # Ensure model dir exists

    # Create dummy config.json for testing purposes (used by the default_cli_config)
    dummy_config_content = {
        "database": {"path": test_db_path},
        "logging": {"file": os.path.basename(test_log_file), "max_log_size_mb": 50},
        "server": {"host": "127.0.0.1", "port": 5000},
        "system": {
            "environment": "development",
            "cpu_usage_threshold": 90,
            "memory_usage_threshold": 90,
            "min_available_memory_gb": 0.5,
            "disk_free_gb_critical": 1.0,
            "disk_free_gb_warning": 5.0,
            "disk_used_percent_warning": 90.0,
            "version": "1.0.0-beta"
        },
        "ml_models": {"model_storage_path": test_lstm_model_dir},
        "analytics": {"api_error_rate_threshold_percent": 10, "api_usage_warning_percent": 90},
    }
    # Write this dummy config to "config.json"
    with open("config.json", "w") as f:
        json.dump(dummy_config_content, f, indent=2)
    print("Created dummy config.json for testing.")

    with open(test_log_file, "w") as f:
        f.write("Test log entry.\n" * 10) # Create a small log file

    # Create a dummy database file for testing DB operations (using the actual path from config)
    if not os.path.exists(test_db_path):
        conn = sqlite3.connect(test_db_path)
        conn.close()

    try:
        main()
    except Exception as e:
        logger.critical(f"System Utilities example failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Final cleanup after testing
        print("\nPerforming final cleanup of test artifacts...")
        if os.path.exists("config.json"): os.remove("config.json")
        if os.path.exists(test_log_file): os.remove(test_log_file)
        if os.path.exists(test_db_path): os.remove(test_db_path)
        if os.path.exists(f"{test_db_path}-wal"): os.remove(f"{test_db_path}-wal")
        if os.path.exists(f"{test_db_path}-shm"): os.remove(f"{test_db_path}-shm")
        if os.path.exists(test_log_dir): shutil.rmtree(test_log_dir)
        if os.path.exists(test_backup_dir): shutil.rmtree(test_backup_dir)
        if os.path.exists(test_lstm_model_dir): shutil.rmtree(test_lstm_model_dir) # Clean model dir
        print("Final cleanup complete.")