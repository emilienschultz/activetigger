import classNames from 'classnames';
import { FC } from 'react';
import { FaCloudDownloadAlt } from 'react-icons/fa';
import { FaGear } from 'react-icons/fa6';
import { MdOutlineTransform } from 'react-icons/md';
import { MdModelTraining } from 'react-icons/md';
import { PiTagDuotone } from 'react-icons/pi';
import { TbListSearch } from 'react-icons/tb';
import { Link } from 'react-router-dom';

import { ProjectStateModel } from '../../types';
import { PossibleProjectActions } from './ProjectPageLayout';

/* define a component for project action bar 
with the project & the current action*/
export const ProjectActionsSidebar: FC<{
  projectState: ProjectStateModel;
  currentProjectAction?: PossibleProjectActions;
  currentScheme?: string;
  currentUser: string;
}> = ({ currentProjectAction, projectState, currentUser }) => {
  const projectName = projectState.params.project_slug;

  // test if computation is currently undergoing
  const currentComputation =
    currentUser in projectState.bertmodels.training ||
    currentUser in projectState.simplemodel.training ||
    (projectState.features.training as string[]).length > 0;

  return (
    <div className="project-sidebar d-flex flex-column flex-shrink-0 p-3 bg-light">
      <ul className="nav nav-pills flex-column mb-auto">
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}`}
            className={classNames('nav-link', !currentProjectAction && 'active')}
            aria-current="page"
          >
            Project{' '}
            <span>
              <b>{projectName}</b>
            </span>
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/features`}
            className={classNames('nav-link', currentProjectAction === 'features' && 'active')}
            aria-current="page"
          >
            <MdOutlineTransform /> Features
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/explorate`}
            className={classNames('nav-link', currentProjectAction === 'explorate' && 'active')}
            aria-current="page"
          >
            <TbListSearch /> Explorate
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/annotate`}
            className={classNames('nav-link', currentProjectAction === 'annotate' && 'active')}
            aria-current="page"
          >
            <PiTagDuotone /> Annotate
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/train`}
            className={classNames('nav-link', currentProjectAction === 'train' && 'active')}
            aria-current="page"
          >
            <MdModelTraining /> Train model
          </Link>
        </li>
        <li className="nav-item">
          <Link
            to={`/projects/${projectName}/export`}
            className={classNames('nav-link', currentProjectAction === 'export' && 'active')}
            aria-current="page"
          >
            <FaCloudDownloadAlt /> Export
          </Link>
        </li>
        {currentComputation && (
          <li className="nav-item ">
            <div className="nav-link computing">
              <span className="computing">
                <FaGear size={20} /> Computing
              </span>
            </div>
          </li>
        )}
      </ul>
    </div>
  );
};
