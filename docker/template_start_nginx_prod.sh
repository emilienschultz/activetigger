#!/usr/bin/env bash
envsubst < /etc/nginx/user_conf.d/nginx.prod.template > /etc/nginx/user_conf.d/default.conf
/scripts/start_nginx_certbot.sh