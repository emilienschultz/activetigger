import { FC, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';

import { useGetProject } from '../core/api';
import { ProjectModel } from '../types';
import { PageLayout } from './layout/PageLayout';

export const ProjectPage: FC = () => {
  const { projectName } = useParams();
  const getProject = useGetProject();
  const [project, setProject] = useState<ProjectModel | undefined>(undefined);

  useEffect(() => {
    if (projectName) getProject(projectName).then((project) => setProject(project));
  }, [getProject, projectName]);

  return <PageLayout>{project && <div>{JSON.stringify(project, null, 2)}</div>}</PageLayout>;
};
