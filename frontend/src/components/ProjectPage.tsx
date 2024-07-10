import { FC } from 'react';
import { useParams } from 'react-router-dom';

import { useProject } from '../core/api';
import { PageLayout } from './layout/PageLayout';

export const ProjectPage: FC = () => {
  const { projectName } = useParams();

  const project = useProject(projectName);

  return <PageLayout>{project && <div>{JSON.stringify(project, null, 2)}</div>}</PageLayout>;
};
