import { FC } from 'react';
import { useParams } from 'react-router-dom';

import { ProjectPageLayout } from './layout/ProjectPageLayout';

export const ProjectAnnotationPage: FC = () => {
  const { projectName } = useParams();

  // we must get the project annotation payload /element
  if (!projectName) return null;
  return (
    <ProjectPageLayout projectName={projectName} currentAction="annotate">
      Annotation {projectName}
    </ProjectPageLayout>
  );
};
