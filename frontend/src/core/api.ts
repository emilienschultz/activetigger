import { values } from 'lodash';
import createClient from 'openapi-fetch';
import { useCallback } from 'react';

import type { components, paths } from '../generated/openapi';
import { AvailableProjectsModel, ProjectDataModel } from '../types';
import { getAuthHeaders, useAuth } from './auth';
import config from './config';
import { getAsyncMemoData, useAsyncMemo } from './useAsyncMemo';

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

export async function me(token: string) {
  const res = await api.GET('/users/me', {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  });

  if (res.data) return res.data;
  //else throw new Error(res.error.detail?.map((d) => d.msg).join('; '));
}

export function useUserProjects(): AvailableProjectsModel[] | undefined {
  const { authenticatedUser } = useAuth();

  const projects = useAsyncMemo(async () => {
    const authHeaders = getAuthHeaders(authenticatedUser);
    if (authHeaders) {
      const res = await api.GET('/projects', {
        ...authHeaders,
        params: {
          header: { username: authHeaders.headers.username },
        },
      });

      // TODO: type API response in Python code
      if (res.data && !res.error)
        return values(res.data.projects) as unknown as AvailableProjectsModel[];
      else throw new Error(res.error.detail?.map((d) => d.msg).join('; '));
    }
    //TODO notify must be loged in
  }, [authenticatedUser]);

  return getAsyncMemoData(projects);
}

export function useCreateProject() {
  const { authenticatedUser } = useAuth();
  const createProject = useCallback(
    async (project: ProjectDataModel) => {
      const authHeaders = getAuthHeaders(authenticatedUser);
      if (authenticatedUser) {
        const res = await api.POST('/projects/new', {
          ...authHeaders,
          params: { header: { username: authenticatedUser.username } },
          body: project,
        });
        if (res.error)
          throw new Error(
            res.error.detail
              ? res.error.detail?.map((d) => d.msg).join('; ')
              : res.error.toString(),
          );
      }
      //TODO: notify
    },
    [authenticatedUser],
  );
  return createProject;
}

export function useProject(projectName?: string) {
  const { authenticatedUser } = useAuth();

  const project = useAsyncMemo(async () => {
    const authHeaders = getAuthHeaders(authenticatedUser);
    if (authenticatedUser && projectName) {
      const res = await api.GET('/state/{project_name}', {
        ...authHeaders,
        params: {
          header: { username: authenticatedUser.username },
          path: { project_name: projectName },
        },
      });
      if (res.error)
        throw new Error(
          res.error.detail ? res.error.detail?.map((d) => d.msg).join('; ') : res.error.toString(),
        );
      return res.data.params;
    }
    //TODO: notify
  }, [authenticatedUser, projectName]);

  return getAsyncMemoData(project);
}
