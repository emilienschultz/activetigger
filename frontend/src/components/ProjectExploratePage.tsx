import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';

import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to display the features page
 */

export const ProjectExploratePage: FC = () => {
  const { projectName } = useParams();
  if (!projectName) return null;
  return (
    <ProjectPageLayout projectName={projectName} currentAction="explorate">
      <div className="container-fluid">
        <div className="row">
          <h2 className="subsection">Data exploration</h2>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
