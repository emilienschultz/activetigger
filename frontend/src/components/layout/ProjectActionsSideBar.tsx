import classNames from 'classnames';
import { FC } from 'react';
import { FaClipboardCheck, FaCloudDownloadAlt } from 'react-icons/fa';
import { MdModelTraining, MdOutlineHomeMax, MdOutlineTransform } from 'react-icons/md';
import { PiTagDuotone } from 'react-icons/pi';
import { TbListSearch } from 'react-icons/tb';
import { Link } from 'react-router-dom';
import { useGetQueue } from '../../core/api';
import { ProjectStateModel } from '../../types';
import { PossibleProjectActions } from './ProjectPageLayout';

/* define a component for project action bar 
with the project & the current action*/
export const ProjectActionsSidebar: FC<{
  projectState: ProjectStateModel | null;
  currentProjectAction?: PossibleProjectActions;
  currentScheme?: string;
  currentUser: string;
}> = ({ currentProjectAction, projectState, currentUser }) => {
  const projectName = projectState ? projectState.params.project_slug : null;

  // test if computation is currently undergoing
  const currentComputation = projectState
    ? currentUser in projectState.bertmodels.training ||
      currentUser in projectState.simplemodel.training ||
      (projectState.features.training as string[]).length > 0
    : false;

  // display the number of current processes on the server
  const { queueState } = useGetQueue(projectState || null);

  return (
    <div className="project-sidebar d-flex flex-column flex-shrink-0 bg-light">
      <ul className="nav nav-pills flex-column mb-auto">
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}`}
            className={classNames('nav-link', !currentProjectAction && 'active')}
            aria-current="page"
            title="Project"
          >
            <MdOutlineHomeMax className="m-2" />
            <span>Project {projectName}</span>
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/features`}
            className={classNames('nav-link', currentProjectAction === 'features' && 'active')}
            aria-current="page"
            title="Features"
          >
            <MdOutlineTransform />
            <span> Features</span>
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/explorate`}
            className={classNames('nav-link', currentProjectAction === 'explorate' && 'active')}
            aria-current="page"
            title="Exploration"
          >
            <TbListSearch />
            <span> Exploration</span>
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/annotate`}
            className={classNames('nav-link', currentProjectAction === 'annotate' && 'active')}
            aria-current="page"
            title="Annotation"
          >
            <PiTagDuotone />
            <span> Annotation</span>
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/train`}
            className={classNames('nav-link', currentProjectAction === 'train' && 'active')}
            aria-current="page"
            title="Training"
          >
            <MdModelTraining />
            <span> Training</span>
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/test`}
            className={classNames('nav-link', currentProjectAction === 'test' && 'active')}
            aria-current="page"
            title="Test"
          >
            <FaClipboardCheck />
            <span> Test</span>
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/export`}
            className={classNames('nav-link', currentProjectAction === 'export' && 'active')}
            aria-current="page"
            title="Export"
          >
            <FaCloudDownloadAlt />
            <span> Export</span>
          </Link>
        </li>
        <li className="nav-item ">
          <div className="nav-link">
            <span className="badge text-bg-info">
              <span className="d-none d-md-inline">Server load: </span>
              {Object.values(queueState || []).length}
            </span>
          </div>
        </li>
        {currentComputation && (
          <li className="nav-item ">
            <div className="nav-link">
              <div className="d-flex justify-content-left align-items-center">
                <div className="spinner-border spinner-border-sm text-warning" role="status">
                  <span className="visually-hidden">Computing</span>
                </div>
                <span className="computing d-none d-md-inline">Computing</span>
              </div>
            </div>
          </li>
        )}
      </ul>
    </div>
  );
};
