import createClient from 'openapi-fetch';

import type { components, paths } from '../generated/openapi';
import config from './config';

const api = createClient<paths>({ baseUrl: `${config.api.url}` });

export type LoginParams = components['schemas']['Body_login_for_access_token_token_post'];

export async function login(params: LoginParams) {
  const res = await api.POST('/token', {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: params,
    bodySerializer: (body) => new URLSearchParams(body as Record<string, string>),
  });

  if (res.data && !res.error) return res.data;
  else throw new Error(res.error.detail?.map((d) => d.msg).join('; '));
}
