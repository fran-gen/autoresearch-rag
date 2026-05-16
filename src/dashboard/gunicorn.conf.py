# gunicorn.conf.py - Security-hardened HTTP/2 configuration

# Limit concurrent streams to prevent resource exhaustion
http2_max_concurrent_streams = 100

# Limit header size to prevent HPACK bomb attacks
http2_max_header_list_size = 65536  # 64KB

# Standard frame size (RFC minimum)
http2_max_frame_size = 16384

# Reasonable flow control window
http2_initial_window_size = 65535  # 64KB

# Connection timeouts to prevent slow attacks
timeout = 30
keepalive = 120
graceful_timeout = 30

# Limit request sizes
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190
