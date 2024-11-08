import { FC, useEffect } from 'react';
import { useParams } from 'react-router-dom';

import { useProject } from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';

/**
 * Component to actualise project state in the context
 */
export const CurrentProjectMonitoring: FC = () => {
  const { projectName } = useParams();
  const { setAppContext, appContext, resetContext } = useAppContext();
  const { authenticatedUser } = useAuth();

  const { project, reFetch } = useProject(projectName); // api call

  // reset context if project change
  useEffect(() => {
    if (projectName != appContext.currentProject?.params.project_slug) {
      console.log('PROJECT CHANGED');
      resetContext();
    }
  }, [projectName, appContext.currentProject?.params.project_slug, resetContext]);

  useEffect(() => {
    if (project && !('detail' in project)) {
      // fix BUG
      setAppContext((prev) => ({ ...prev, currentProject: project }));
    }

    // check if training process, and refresh the value
    const isComputing =
      project &&
      authenticatedUser &&
      Object.keys(project.bertmodels.training).includes(authenticatedUser.username);

    if (typeof isComputing === 'boolean')
      setAppContext((prev) => ({ ...prev, isComputing: isComputing }));
  }, [project, setAppContext, authenticatedUser]);

  // Effect to poll project data regularly to monitor long lasting server tasks
  // each time reFetch change
  useEffect(() => {
    //expose refetch method into context
    setAppContext((prev) => ({ ...prev, reFetchCurrentProject: reFetch }));
    // execute a fetch call to update project data every 2000ms
    const intervalId = setInterval(reFetch, 3000);
    // useEffect can return a method which is executed when the component is unmounted
    return () => {
      clearInterval(intervalId);
    };
  }, [reFetch, setAppContext]);

  return null;
};
