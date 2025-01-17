import { FC, PropsWithChildren } from 'react';

import { useAuth } from '../../core/auth';
import { useAppContext } from '../../core/context';
import { PageLayout } from './PageLayout';
import { ProjectActionsSidebar } from './ProjectActionsSideBar';

export type PossibleProjectActions =
  | 'annotate'
  | 'train'
  | 'parameters'
  | 'prepare'
  | 'explore'
  | 'curate'
  | 'monitor'
  | 'test'
  | 'generate'
  | 'export'
  | 'predict';

type ProjectPageLayoutProps = PropsWithChildren<{
  projectName: string | null;
  currentAction?: PossibleProjectActions;
}>;

/* On a specific project, add the ActionSideBar*/
export const ProjectPageLayout: FC<ProjectPageLayoutProps> = ({
  projectName,
  currentAction,
  children,
}) => {
  // get the current state of the project
  const {
    appContext: { currentProject: project, currentScheme, phase, developmentMode },
  } = useAppContext();

  // get current user
  const { authenticatedUser } = useAuth();
  if (!authenticatedUser) return '';

  return (
    <PageLayout currentPage="projects" projectName={projectName || null}>
      <div className="container-fluid">
        <div className="row">
          <div className="col-1 col-md-3 col-lg-2">
            <ProjectActionsSidebar
              projectState={project || null}
              currentProjectAction={currentAction}
              currentScheme={currentScheme}
              currentUser={authenticatedUser.username}
              currentMode={phase}
              developmentMode={developmentMode}
            />
          </div>
          <div className="col-11 col-md-9 col-lg-10">{children}</div>
        </div>
      </div>
    </PageLayout>
  );
};
