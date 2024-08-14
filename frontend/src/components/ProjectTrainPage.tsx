import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';

import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to manage model training
 */

export const ProjectTrainPage: FC = () => {
  const { projectName } = useParams();
  if (!projectName) return null;
  return (
    <ProjectPageLayout projectName={projectName} currentAction="train">
      <div className="container-fluid">
        <div className="row">
          <h2 className="subsection">Train a bert model</h2>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
