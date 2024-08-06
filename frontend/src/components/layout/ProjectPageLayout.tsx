import { FC, PropsWithChildren } from 'react';

import { PageLayout } from './PageLayout';
import { ProjectActionsSidebar } from './ProjectActionsSideBar';

export type PossibleProjectActions = 'annotate' | 'train' | 'parameters';

type ProjectPageLayoutProps = PropsWithChildren<{
  projectName: string;
  currentAction?: PossibleProjectActions;
}>;

/* On a specific project, add the ActionSideBar*/
export const ProjectPageLayout: FC<ProjectPageLayoutProps> = ({
  projectName,
  currentAction,
  children,
}) => {
  return (
    <PageLayout currentPage="projects" projectName={projectName}>
      <div className="container-fluid">
        <div className="row">
          <div className="col-3">
            <ProjectActionsSidebar projectName={projectName} currentProjectAction={currentAction} />
          </div>
          <div className="col-9">{children}</div>
        </div>
      </div>
    </PageLayout>
  );
};
