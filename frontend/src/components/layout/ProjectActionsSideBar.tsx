import classNames from 'classnames';
import { FC } from 'react';
import { FaClipboardCheck, FaCloudDownloadAlt } from 'react-icons/fa';
import { MdModelTraining, MdOutlineHomeMax, MdOutlineTransform } from 'react-icons/md';
import { PiTagDuotone } from 'react-icons/pi';
import { RiAiGenerate } from 'react-icons/ri';
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
  currentMode?: string;
  currentScheme?: string;
  currentUser: string;
}> = ({ currentProjectAction, projectState, currentUser, currentMode, currentScheme }) => {
  const projectName = projectState ? projectState.params.project_slug : null;

  // test if computation is currently undergoing
  const currentComputation = projectState
    ? currentUser in projectState.bertmodels.training ||
      currentUser in projectState.simplemodel.training ||
      currentUser in projectState.projections.training ||
      (projectState.features.training as string[]).length > 0
    : false;

  // display the number of current processes on the server
  const { queueState, gpu } = useGetQueue(projectState || null);

  return (
    <div
      className={`project-sidebar d-flex flex-column flex-shrink-0 ${currentMode == 'train' ? 'bg-light' : 'bg-warning'}`}
    >
      <ul className="nav nav-pills flex-column mb-auto">
        <li className="nav-item mt-3">
          <Link
            to={`/projects/${projectName}`}
            className={classNames('nav-link', !currentProjectAction && 'active')}
            aria-current="page"
            title="Project"
          >
            <MdOutlineHomeMax className="m-2" />
            <span>
              <b>{projectName}</b>
            </span>
            <span className="mx-2" style={{ fontSize: '0.875rem', color: 'grey' }}>
              {currentScheme}
            </span>
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/prepare`}
            className={classNames('nav-link', currentProjectAction === 'prepare' && 'active')}
            aria-current="page"
            title="Prepare"
          >
            <MdOutlineTransform />
            <span> Prepare</span>
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/explore`}
            className={classNames('nav-link', currentProjectAction === 'explore' && 'active')}
            aria-current="page"
            title="Explore"
          >
            <TbListSearch />
            <span> Explore</span>
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
            <span> Annotate</span>
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
            <span> Train</span>
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
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/generate`}
            className={classNames('nav-link', currentProjectAction === 'generate' && 'active')}
            aria-current="page"
            title="Generate"
          >
            <RiAiGenerate />
            <span> Generate</span>
          </Link>
        </li>
        <li className="nav-item ">
          <div className="nav-link">
            <div className="badge text-bg-info">
              <span className="d-none d-md-inline">Server load: </span>
              {Object.values(queueState || []).length}
            </div>
            <br></br>
            <div className="badge text-bg-warning" title="Available GPU memory">
              <span className="d-none d-md-inline">
                GPU:
                {gpu
                  ? `${gpu['total_memory'] - gpu['available_memory']} / ${gpu['total_memory']} Go`
                  : 'No'}
              </span>
            </div>
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
