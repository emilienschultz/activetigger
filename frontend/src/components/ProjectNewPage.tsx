import { FC } from 'react';

import { ProjectCreationForm } from './forms/ProjectCreationForm';
import { PageLayout } from './layout/PageLayout';

export const ProjectNewPage: FC = () => {
  return (
    <PageLayout>
      <ProjectCreationForm />
    </PageLayout>
  );
};
