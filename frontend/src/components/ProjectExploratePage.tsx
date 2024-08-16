import { ColDef, ColGroupDef } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import { AgGridReact } from 'ag-grid-react';
import { FC, useState } from 'react';
import { useParams } from 'react-router-dom';

import { useGetTableElements } from '../core/api';
import { useAppContext } from '../core/context';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to display the features page
 */

export const ProjectExploratePage: FC = () => {
  const { projectName } = useParams();
  if (!projectName) return null;
  const {
    appContext: { currentScheme },
    setAppContext,
  } = useAppContext();
  if (!currentScheme) return null;
  const [rowData, setRowData] = useState([]);
  const [page, setPage] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [loading, setLoading] = useState(false);
  const pageSize = 10;
  const { table } = useGetTableElements(
    projectName,
    currentScheme,
    page * pageSize,
    (page + 1) * pageSize,
  );
  console.log(table);

  const onPaginationChanged = (event: any) => {
    if (event.api.paginationGetCurrentPage() !== page) {
      setPage(event.api.paginationGetCurrentPage());
    }
  };

  const columnDefs: ColDef[] = [
    { field: 'index', headerName: 'ID', sortable: true, filter: true },
    { field: 'timestamp', headerName: 'Timestamp', sortable: true, filter: true },
    { field: 'labels', headerName: 'Label', sortable: true, filter: true },
    { field: 'text', headerName: 'Text', sortable: true, filter: true },
  ];

  //Je veux utiliser AG Grid https://www.ag-grid.com/react-data-grid/getting-started/?utm_source=ag-grid-react-readme&utm_medium=repository&utm_campaign=github pour afficher un tableau en react des éléments de ma base de données, paginées par 10, avec un appel à l'API pour charger par page de 10

  return (
    <ProjectPageLayout projectName={projectName} currentAction="explorate">
      <div className="container-fluid">
        <div className="row">
          <div className="col-0"></div>
          <div className="col-10">
            <h2 className="subsection">Data exploration</h2>
            <div
              className="ag-theme-quartz" // applying the Data Grid theme
              style={{ height: 800 }}
            >
              {table && (
                <AgGridReact
                  rowData={table['items'] ? table['items'] : []}
                  columnDefs={columnDefs}
                  pagination={true}
                  paginationPageSize={pageSize}
                  onPaginationChanged={onPaginationChanged}
                  suppressPaginationPanel={false}
                  loadingOverlayComponentParams={{ loadingMessage: 'Chargement...' }}
                  overlayLoadingTemplate='<span class="ag-overlay-loading-center">Chargement des données...</span>'
                  overlayNoRowsTemplate='<span class="ag-overlay-no-rows-center">Aucune donnée disponible</span>'
                />
              )}
            </div>{' '}
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
